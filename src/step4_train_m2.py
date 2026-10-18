#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 4 -- Entraînement de M2 (latéralisation : 0 = gauche, 1 = droite, 2 = bilatéral) sur les fenêtres des événements de crise.

Données : cache de l'étape 2b, patients séparés par l'étape 2a. `eval` n'est JAMAIS lue.
Choix logiques (leçons de M1)
  * Normalisation (--norm) :
      window       : par canal et par fenêtre -> perd l'amplitude absolue ET les rapports entre canaux (donc l'asymétrie gauche/droite)
      windowglobal : un seul écart-type par fenêtre -> conserve les rapports entre canaux, perd l'amplitude absolue
      const        : division par 30 µV -> conserve tout
    On les compare, au lieu d'en supposer une : les trois sont des essais à faire.
  * Tirage : 3 classes équiprobables ; dans une classe, patient tiré ~ (nombre de fenêtres)^0,5.
  * Miroir gauche/droite : les canaux sont échangés ET les labels gauche <-> droite sont échangés (bilatéral inchangé).
    C'est valide parce que la règle des labels est symétrique (égalité -> bilatéral).
  * Sélection : exactitude équilibrée par fenêtre, pondérée par patient. On rapporte aussi le niveau ÉVÉNEMENT (probabilités
    moyennes des fenêtres d'un même événement), qui est l'unité d'évaluation finale.

Usage :  python step4_train_m2.py --smoke   |   --norm const --seed 0 --out_dir .../m2_const_seed0   |   --selftest
"""
import argparse, os, sys, json, time, random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_train_m1 import load_cache, normalize_batch, FLIP_PERM, BalancedSampler

CLASSES = ["gauche", "droite", "bilatéral"]
FLIP_LABEL = np.array([1, 0, 2])          # miroir : gauche <-> droite, bilatéral inchangé


def m2_indices(D, role):
    return np.where((D["role"] == role) & (D["m2"] >= 0))[0]


def m2_metrics(y, pred, pid):
    """Exactitude, rappels par classe, exactitude équilibrée, et versions pondérées par patient (chaque patient pèse pareil
    dans chaque classe). Une classe absente donne NaN et est ignorée de la moyenne."""
    y, pred, pid = np.asarray(y), np.asarray(pred), np.asarray(pid)
    key = pd.Series(pid).astype(str) + "_" + pd.Series(y).astype(str)
    w = 1.0 / key.map(key.value_counts()).values
    rec, rec_pw = [], []
    for c in range(3):
        m = y == c
        rec.append(float((pred[m] == c).mean()) if m.any() else np.nan)
        rec_pw.append(float(np.average(pred[m] == c, weights=w[m])) if m.any() else np.nan)
    conf = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    return dict(acc=float((pred == y).mean()), bacc=float(np.nanmean(rec)), bacc_pw=float(np.nanmean(rec_pw)),
                recall=rec, recall_pw=rec_pw, confusion=conf, n=len(y))


def event_aggregate(rec, eid, pid, y, probs):
    """Un événement = (enregistrement, indice d'événement). Prédiction = classe de plus forte probabilité MOYENNE."""
    df = pd.DataFrame(dict(rec=rec, eid=eid, pid=pid, y=y, pL=probs[:, 0], pR=probs[:, 1], pB=probs[:, 2]))
    g = df.groupby(["rec", "eid"]).agg(y=("y", "first"), pid=("pid", "first"), pL=("pL", "mean"), pR=("pR", "mean"),
                                       pB=("pB", "mean"), n=("y", "size")).reset_index()
    g["pred"] = g[["pL", "pR", "pB"]].values.argmax(1)
    return g


def fmt_metrics(m, tag):
    r = m["recall"]
    return (f"{tag}: exactitude {m['acc']:.3f} | équilibrée {m['bacc']:.3f} (par patient {m['bacc_pw']:.3f}) | rappels G/D/B "
            f"{r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f} | n={m['n']}")


def predict_proba_m2(model, x, idx, device, bs=256, mode="const"):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb = torch.from_numpy(x[idx[i:i + bs]]).to(device).float()
            out.append(torch.softmax(model(normalize_batch(xb, mode)), dim=1).cpu().numpy())
    return np.concatenate(out)


def train(args):
    import torch, torch.nn as nn
    from eeg_model import EEGConformer
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device, "| torch", torch.__version__, flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    D = load_cache(args.cache_dir, args.split_csv, limit_per_role=args.limit_per_role, seed=args.seed, verbose=False)
    itr, itv = m2_indices(D, "train"), m2_indices(D, "val")
    for ro, ii in (("train", itr), ("val", itv)):
        print(f"{ro}: {len(ii)} fenêtres M2, {len(set(D['pid'][ii]))} patients, G/D/B " + "/".join(str(int((D['m2'][ii] == c).sum())) for c in range(3)), flush=True)
    sampler = BalancedSampler(D["m2"][itr], D["pid"][itr], seed=args.seed, n_classes=3)
    model = EEGConformer(n_canaux=18, n_classes=3, use_attn=not args.no_attn).to(device)
    print("paramètres :", sum(p.numel() for p in model.parameters()), "| normalisation :", args.norm, flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    perm = torch.tensor(FLIP_PERM, device=device)
    flip_lab = torch.tensor(FLIP_LABEL, device=device)
    best, best_ep, bad, log = -1.0, -1, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            gi = itr[sampler.draw(args.batch)]
            xb = normalize_batch(torch.from_numpy(D["x"][gi]).to(device).float(), args.norm)
            yb = torch.from_numpy(D["m2"][gi]).to(device)
            if args.flip:
                sel = torch.where(torch.rand(len(xb), device=device) < 0.5)[0]
                if len(sel):
                    xb[sel] = xb[sel][:, perm]
                    yb[sel] = flip_lab[yb[sel]]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        P = predict_proba_m2(model, D["x"], itv, device, mode=args.norm)
        mw = m2_metrics(D["m2"][itv], P.argmax(1), D["pid"][itv])
        ev = event_aggregate(D["rec"][itv], D["eid"][itv], D["pid"][itv], D["m2"][itv], P)
        me = m2_metrics(ev.y.values, ev.pred.values, ev.pid.values)
        row = dict(epoch=ep + 1, train_loss=float(np.mean(losses)), win_acc=mw["acc"], win_bacc=mw["bacc"], win_bacc_pw=mw["bacc_pw"],
                   ev_acc=me["acc"], ev_bacc=me["bacc"], ev_bacc_pw=me["bacc_pw"], seconds=round(time.time() - t0))
        log.append(row)
        pd.DataFrame(log).to_csv(os.path.join(args.out_dir, "m2_log.csv"), index=False)
        flag = ""
        if mw["bacc_pw"] > best:
            best, best_ep, bad = mw["bacc_pw"], ep + 1, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "m2_best.pt"))
            flag = "  <- meilleur"
        else:
            bad += 1
        print(f"ep {ep + 1:2d} | perte {row['train_loss']:.4f} | fenêtres : exact. {mw['acc']:.3f}, équilibrée {mw['bacc']:.3f} (patient {mw['bacc_pw']:.3f}) | "
              f"événements : exact. {me['acc']:.3f}, équilibrée {me['bacc']:.3f} | {row['seconds']} s{flag}", flush=True)
        if bad >= args.patience:
            print("arrêt anticipé", flush=True)
            break
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "m2_best.pt"), map_location=device))
    P = predict_proba_m2(model, D["x"], itv, device, mode=args.norm)
    pd.DataFrame(dict(recording=D["rec_names"][D["rec"][itv]], eid=D["eid"][itv], start_sample=D["start"][itv], patient=D["patients"][D["pid"][itv]],
                      y=D["m2"][itv], pL=P[:, 0], pR=P[:, 1], pB=P[:, 2])).to_csv(os.path.join(args.out_dir, "m2_val_scores.csv"), index=False)
    mw = m2_metrics(D["m2"][itv], P.argmax(1), D["pid"][itv])
    ev = event_aggregate(D["rec"][itv], D["eid"][itv], D["pid"][itv], D["m2"][itv], P)
    me = m2_metrics(ev.y.values, ev.pred.values, ev.pid.values)
    with open(os.path.join(args.out_dir, "m2_config.json"), "w") as f:
        json.dump(dict(vars(args), torch=torch.__version__, best_epoch=best_ep, win=mw | dict(confusion=mw["confusion"].tolist()),
                       ev=me | dict(confusion=me["confusion"].tolist()), majority_rate_events=float(max(np.bincount(ev.y.values, minlength=3)) / len(ev))), f, indent=2)
    print("<<<COPY_FROM_HERE>>>")
    print(f"M2 seed {args.seed} | norm {args.norm} | meilleure époque {best_ep} | durée {round(time.time() - t0)} s")
    print(fmt_metrics(mw, "validation, FENÊTRES"))
    print(fmt_metrics(me, "validation, ÉVÉNEMENTS"))
    print("taux de la classe majoritaire (événements de validation) :", f"{max(np.bincount(ev.y.values, minlength=3)) / len(ev):.3f}")
    print("matrice de confusion ÉVÉNEMENTS (lignes = vrai G/D/B, colonnes = prédit G/D/B) :", me["confusion"].tolist())
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test (numpy)
def selftest():
    import tempfile, shutil
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    check("FLIP_LABEL : involution, bilatéral inchangé", FLIP_LABEL[FLIP_LABEL].tolist() == [0, 1, 2] and FLIP_LABEL[2] == 2)
    y = np.array([0, 0, 1, 1, 2, 2])
    m = m2_metrics(y, y, np.arange(6))
    check("m2_metrics : prédictions parfaites -> 1", m["acc"] == 1 and m["bacc"] == 1 and m["bacc_pw"] == 1 and m["confusion"].trace() == 6)
    # un gros patient prédit parfaitement, un petit patient toujours faux : l'exactitude globale est élevée, celle par patient non
    yb = np.array([0] * 50 + [0] * 2)
    pb = np.array([0] * 50 + [1] * 2)
    pidb = np.array([0] * 50 + [1] * 2)
    mm = m2_metrics(np.r_[yb, [1, 2]], np.r_[pb, [1, 2]], np.r_[pidb, [2, 3]])
    check(f"m2_metrics : rappel gauche global {mm['recall'][0]:.2f} > rappel par patient {mm['recall_pw'][0]:.2f}", mm["recall"][0] > mm["recall_pw"][0])
    mc = m2_metrics(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0, 1, 2]))
    check("m2_metrics : classe absente ignorée (NaN) sans erreur", np.isnan(mc["recall"][2]) and mc["bacc"] == 1.0)
    # agrégation par événement
    probs = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.2, 0.2, 0.6], [0.6, 0.3, 0.1]])
    ev = event_aggregate(np.array(["r1", "r1", "r1", "r1", "r2"]), np.array([0, 0, 1, 1, 0]), np.array([0, 0, 0, 0, 1]), np.array([0, 0, 2, 2, 0]), probs)
    check("event_aggregate : 3 événements, moyenne des probabilités (0,55 / 0,375 / 0,05 -> gauche ; bilatéral ; gauche)",
          len(ev) == 3 and ev.sort_values(["rec", "eid"]).pred.tolist() == [0, 2, 0])
    # sampler 3 classes + sélection M2
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "cache", "train"))
        rows, sp = [], []
        valid = np.ones(18, bool)
        for i in range(18):
            pat = f"p{i // 3:02d}"
            n = 12
            m2 = np.array([-1] * 3 + [i % 3] * 9, np.int8)          # chaque enregistrement : une classe M2
            eid = np.array([-1] * 3 + [0] * 9, np.int16)
            x = (rng.standard_normal((n, 18, 1000)) * 15).astype(np.float16)
            np.savez(os.path.join(tmp, "cache", "train", f"{pat}_s001_t00{i % 3}.npz"), x=x, start_sample=np.arange(n) * 500,
                     m1=np.array([0] * 3 + [1] * 9, np.int8), m2=m2, eid=eid, valid=valid)
            rows.append(dict(split="train", recording=f"{pat}_s001_t00{i % 3}", patient=pat, status="ok"))
        for pat in sorted({r["patient"] for r in rows}):
            sp.append(dict(patient=pat, role="val" if pat == "p05" else "train"))
        pd.DataFrame(rows).to_csv(os.path.join(tmp, "cache", "manifest.csv"), index=False)
        pd.DataFrame(sp).to_csv(os.path.join(tmp, "split.csv"), index=False)
        D = load_cache(os.path.join(tmp, "cache"), os.path.join(tmp, "split.csv"), verbose=False)
        itr, itv = m2_indices(D, "train"), m2_indices(D, "val")
        check("m2_indices : uniquement des fenêtres M2 (>= 0), par rôle", (D["m2"][itr] >= 0).all() and (D["m2"][itv] >= 0).all() and len(itv) > 0)
        s = BalancedSampler(D["m2"][itr], D["pid"][itr], seed=1, n_classes=3)
        d = s.draw(30000)
        freq = np.bincount(D["m2"][itr][d], minlength=3) / 30000
        check(f"BalancedSampler 3 classes : ~1/3 chacune ({np.round(freq, 3).tolist()})", np.abs(freq - 1 / 3).max() < 0.02)
        check("load_cache renvoie eid", "eid" in D and D["eid"].shape == D["y"].shape)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    print("NB : la partie torch (modèle, boucle) n'est PAS couverte ici : lancer --smoke dans Colab.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--out_dir", default=base + "m2_const_seed0")
    ap.add_argument("--norm", choices=["window", "windowglobal", "const"], default="const")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps_per_epoch", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--no_flip", dest="flip", action="store_false")
    ap.add_argument("--no_attn", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.smoke:
        a.limit_per_role, a.epochs, a.steps_per_epoch, a.batch = 150, 2, 20, 16
        a.out_dir = a.out_dir.rstrip("/") + "_smoke"
    train(a)
