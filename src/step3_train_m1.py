#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 3 -- Entraînement de M1 (détecteur de crise : fond = 0, crise = 1).

Données : le cache de l'étape 2b ; patients séparés en entraînement / validation par l'étape 2a. `eval` n'est JAMAIS lue.
Choix logiques (et leur raison)
  * Fenêtres plates (écart-type médian < 0,5 µV) retirées au chargement.
  * Tirage équilibré : 50 % crise / 50 % fond, et à l'intérieur de chaque classe un patient est tiré avec une probabilité
    proportionnelle à (nombre de fenêtres)^0,5 : un patient très long ne domine plus l'entraînement.
  * Normalisation (--norm) faite sur GPU par lot : "window" (défaut) = par canal et par fenêtre (centrage, division par
    l'écart-type, clip +-10), qui perd l'amplitude absolue ; "const" = division par 30 µV, qui la conserve.
  * Augmentation : miroir gauche/droite (exact avec nos 18 dérivations), une fenêtre sur deux. Le label crise/fond ne change pas.
  * Sélection du meilleur modèle : AUC de validation PONDÉRÉE PAR PATIENT (chaque patient compte autant dans chaque classe).
    L'AUC ne dépend pas d'un seuil ; le seuil est choisi plus tard (étape 5), sur la validation.
  * Graines fixées, configuration et versions enregistrées.

Usage (Colab, GPU) :
    python step3_train_m1.py --smoke        # essai de quelques minutes : vérifie que tout tourne
    python step3_train_m1.py --cache_dir .../cache --split_csv .../etape2a/split_patients.csv --out_dir .../m1_seed0 --seed 0
    python step3_train_m1.py --selftest     # partie numpy uniquement (sans torch)
"""
import argparse, os, sys, json, time, random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2b_cache_windows import flat_window_mask, DERIVATIONS, WIN

FLIP_PERM = list(range(8, 16)) + list(range(0, 8)) + [16, 17]       # échange des deux hémisphères ; FZ-CZ, CZ-PZ inchangés


# ------------------------------------------------------------------ données (numpy)
def load_cache(cache_dir, split_csv, roles=("train", "val"), limit_per_role=None, seed=0, verbose=True):
    man = pd.read_csv(os.path.join(cache_dir, "manifest.csv")).drop_duplicates("recording", keep="last")
    man = man[man.status == "ok"]
    sp = pd.read_csv(split_csv)[["patient", "role"]]
    man = man.merge(sp, on="patient", how="inner")                 # les patients hors découpage (ex. eval) sont ignorés
    man = man[man.role.isin(roles)].sort_values("recording")
    if limit_per_role:
        man = pd.concat([g.sample(min(len(g), limit_per_role), random_state=seed) for _, g in man.groupby("role")]).sort_values("recording")
    xs, ys, m2s, pids, recs, starts, rls, vals, eids = [], [], [], [], [], [], [], [], []
    patients = sorted(man.patient.unique())
    pcode = {p: i for i, p in enumerate(patients)}
    rec_names, n_flat, n_amb = [], 0, 0
    for ri, r in enumerate(man.itertuples()):
        z = np.load(os.path.join(cache_dir, r.split, r.recording + ".npz"))
        x, valid, m1 = z["x"], z["valid"], z["m1"]
        flat = flat_window_mask(x, valid)
        keep = (~flat) & (m1 >= 0)
        n_flat += int(flat.sum())
        n_amb += int(((m1 < 0) & ~flat).sum())
        if not keep.any():
            continue
        k = int(keep.sum())
        xs.append(x[keep]); ys.append(m1[keep].astype(np.int64)); m2s.append(z["m2"][keep].astype(np.int64))
        pids.append(np.full(k, pcode[r.patient])); recs.append(np.full(k, len(rec_names))); rls.append(np.full(k, r.role))
        starts.append(z["start_sample"][keep]); rec_names.append(r.recording)
        vals.append(np.repeat(valid[None, :], k, axis=0)); eids.append(z["eid"][keep].astype(np.int64))
    D = dict(x=np.concatenate(xs), y=np.concatenate(ys), m2=np.concatenate(m2s), pid=np.concatenate(pids), rec=np.concatenate(recs),
             start=np.concatenate(starts), role=np.concatenate(rls), valid=np.concatenate(vals), eid=np.concatenate(eids), patients=np.array(patients), rec_names=np.array(rec_names))
    if verbose:
        print(f"chargé : {len(D['y'])} fenêtres ({n_flat} plates retirées) | " + " | ".join(
            f"{ro}: {int((D['role'] == ro).sum())} fenêtres, {len(set(D['pid'][D['role'] == ro]))} patients, crise {int(((D['role'] == ro) & (D['y'] == 1)).sum())}"
            for ro in roles), flush=True)
    return D


def append_hardneg(D, hardneg_dir, split_csv):
    """Ajoute à D (rôle train) les fenêtres normales difficiles écrites par l'étape 8. Refuse tout patient qui n'est pas du rôle train."""
    man = pd.read_csv(os.path.join(hardneg_dir, "manifest.csv")).drop_duplicates("recording", keep="last")
    man = man[man.status == "ok"]
    man = man.merge(pd.read_csv(split_csv)[["patient", "role"]], on="patient", how="left")
    if (man.role != "train").any():
        raise SystemExit("ERREUR : des fenêtres normales difficiles viennent de patients qui ne sont pas du rôle train : "
                         + ", ".join(sorted(set(man[man.role != "train"].patient))[:5]))
    pcode = {p: i for i, p in enumerate(D["patients"])}
    man = man[man.patient.isin(pcode)].sort_values("recording")
    xs, pids, recs, starts, vals, names = [], [], [], [], [], []
    for r in man.itertuples():
        path = os.path.join(hardneg_dir, r.split, r.recording + ".npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        k = len(z["x"])
        xs.append(z["x"]); pids.append(np.full(k, pcode[r.patient])); recs.append(np.full(k, len(D["rec_names"]) + len(names)))
        starts.append(z["start_sample"]); vals.append(np.repeat(z["valid"][None, :], k, axis=0)); names.append(r.recording + "#hn")
    if not xs:
        return D
    k = sum(len(x) for x in xs)
    D2 = dict(D)
    D2["x"] = np.concatenate([D["x"]] + xs)
    D2["y"] = np.concatenate([D["y"], np.zeros(k, np.int64)])
    D2["m2"] = np.concatenate([D["m2"], np.full(k, -1, np.int64)])
    D2["pid"] = np.concatenate([D["pid"]] + pids)
    D2["rec"] = np.concatenate([D["rec"]] + recs)
    D2["start"] = np.concatenate([D["start"]] + starts)
    D2["role"] = np.concatenate([D["role"], np.full(k, "train")])
    D2["valid"] = np.concatenate([D["valid"]] + vals)
    D2["eid"] = np.concatenate([D["eid"], np.full(k, -1, np.int64)])
    D2["rec_names"] = np.concatenate([D["rec_names"], np.array(names)])
    return D2


class BalancedSampler:
    """Classes équiprobables (50/50 pour M1, 1/3 pour M2) ; dans une classe, patient tiré avec une probabilité ~ n_fenêtres^power ; fenêtre uniforme dans le patient."""
    def __init__(self, y, pid, seed=0, power=0.5, n_classes=2):
        self.rng = np.random.default_rng(seed)
        self.n_classes = n_classes
        df = pd.DataFrame({"i": np.arange(len(y)), "y": y, "p": pid})
        self.arrs, self.prob = {}, {}
        for c in range(n_classes):
            arrs = [g.i.values for _, g in df[df.y == c].groupby("p")]
            if not arrs:
                raise ValueError(f"aucune fenêtre de classe {c} dans l'entraînement")
            n = np.array([len(a) for a in arrs], float) ** power
            self.arrs[c], self.prob[c] = arrs, n / n.sum()

    def draw(self, n):
        cls = self.rng.integers(0, self.n_classes, n)
        out = np.empty(n, np.int64)
        for c in range(self.n_classes):
            pos = np.where(cls == c)[0]
            chosen = self.rng.choice(len(self.arrs[c]), size=len(pos), p=self.prob[c])
            for k, ci in zip(pos, chosen):
                a = self.arrs[c][ci]
                out[k] = a[self.rng.integers(len(a))]
        return out


def val_metrics(y, p, pid, thr=0.5):
    """AUC simple et AUC/sensibilité/spécificité pondérées par patient (chaque patient pèse autant dans chaque classe)."""
    from sklearn.metrics import roc_auc_score
    y, p, pid = np.asarray(y), np.asarray(p, float), np.asarray(pid)
    if len(set(y.tolist())) < 2:
        raise ValueError("val_metrics : les données ne contiennent qu'une seule classe")
    key = pd.Series(pid).astype(str) + "_" + pd.Series(y).astype(str)
    w = 1.0 / key.map(key.value_counts()).values
    pred = p >= thr
    sens = float(np.average(pred[y == 1], weights=w[y == 1]))
    spec = float(np.average(~pred[y == 0], weights=w[y == 0]))
    return dict(auc=float(roc_auc_score(y, p)), auc_pw=float(roc_auc_score(y, p, sample_weight=w)),
                sens_pw=sens, spec_pw=spec, bacc_pw=(sens + spec) / 2)


# ------------------------------------------------------------------ entraînement (torch)
CONST_UV = 30.0       # échelle fixe du mode "const" (µV) ; l'amplitude absolue du signal est alors conservée


def normalize_batch(xb, mode="window"):
    """(B, 18, T) en µV.
    mode "window" : centré, divisé par l'écart-type de chaque canal, clip +-10 (l'amplitude absolue est PERDUE).
    mode "windowglobal" : centré, divisé par UN écart-type pour toute la fenêtre, clip +-10 (les rapports entre canaux sont conservés).
    mode "const"  : centré, divisé par une constante (30 µV), clip +-20 (l'amplitude absolue est CONSERVÉE).
    Un canal nul reste nul dans les deux cas."""
    xb = xb - xb.mean(dim=-1, keepdim=True)
    if mode == "const":
        return (xb / CONST_UV).clamp(-20, 20)
    if mode == "windowglobal":
        sd = (xb ** 2).mean(dim=(-1, -2), keepdim=True).sqrt()
        return (xb / (sd + 1e-8)).clamp(-10, 10)
    sd = (xb ** 2).mean(dim=-1, keepdim=True).sqrt()
    return (xb / (sd + 1e-8)).clamp(-10, 10)


def predict_proba_m1(model, x, idx, device, bs=256, mode="window"):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb = torch.from_numpy(x[idx[i:i + bs]]).to(device).float()
            out.append(torch.softmax(model(normalize_batch(xb, mode)), dim=1)[:, 1].cpu().numpy())
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
    D = load_cache(args.cache_dir, args.split_csv, limit_per_role=args.limit_per_role, seed=args.seed)
    if args.hardneg_dir:
        n0 = len(D["y"])
        D = append_hardneg(D, args.hardneg_dir, args.split_csv)
        print(f"fenêtres normales difficiles/aléatoires ajoutées à l'entraînement : {len(D['y']) - n0} ({args.hardneg_dir})", flush=True)
    itr, itv = np.where(D["role"] == "train")[0], np.where(D["role"] == "val")[0]
    sampler = BalancedSampler(D["y"][itr], D["pid"][itr], seed=args.seed)
    model = EEGConformer(n_canaux=18, n_classes=2, use_attn=not args.no_attn).to(device)
    print("paramètres :", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    perm = torch.tensor(FLIP_PERM, device=device)
    best, best_ep, bad, log = -1.0, -1, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            gi = itr[sampler.draw(args.batch)]
            xb = normalize_batch(torch.from_numpy(D["x"][gi]).to(device).float(), args.norm)
            yb = torch.from_numpy(D["y"][gi]).to(device)
            if args.flip:
                sel = torch.where(torch.rand(len(xb), device=device) < 0.5)[0]
                if len(sel):
                    xb[sel] = xb[sel][:, perm]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        p = predict_proba_m1(model, D["x"], itv, device, mode=args.norm)
        met = val_metrics(D["y"][itv], p, D["pid"][itv])
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        vloss = float(-np.mean(D["y"][itv] * np.log(pc) + (1 - D["y"][itv]) * np.log(1 - pc)))
        row = dict(epoch=ep + 1, train_loss=float(np.mean(losses)), val_loss=vloss, **met, seconds=round(time.time() - t0))
        log.append(row)
        pd.DataFrame(log).to_csv(os.path.join(args.out_dir, "m1_log.csv"), index=False)
        flag = ""
        if met["auc_pw"] > best:
            best, best_ep, bad = met["auc_pw"], ep + 1, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "m1_best.pt"))
            flag = "  <- meilleur"
        else:
            bad += 1
        print(f"ep {ep + 1:2d} | perte {row['train_loss']:.4f} | val AUC {met['auc']:.4f} | val AUC par patient {met['auc_pw']:.4f} | "
              f"sens/spéc (patient, seuil 0,5) {met['sens_pw']:.3f}/{met['spec_pw']:.3f} | {row['seconds']} s{flag}", flush=True)
        if bad >= args.patience:
            print("arrêt anticipé", flush=True)
            break
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "m1_best.pt"), map_location=device))
    p = predict_proba_m1(model, D["x"], itv, device, mode=args.norm)
    pd.DataFrame(dict(recording=D["rec_names"][D["rec"][itv]], start_sample=D["start"][itv], patient=D["patients"][D["pid"][itv]],
                      y=D["y"][itv], p=p)).to_csv(os.path.join(args.out_dir, "m1_val_scores.csv"), index=False)
    met = val_metrics(D["y"][itv], p, D["pid"][itv])
    cfg = dict(vars(args), torch=torch.__version__, numpy=np.__version__, best_epoch=best_ep, n_train=int(len(itr)), n_val=int(len(itv)),
               best_val=met)
    with open(os.path.join(args.out_dir, "m1_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("<<<COPY_FROM_HERE>>>")
    print(f"M1 seed {args.seed} | fenêtres entraînement {len(itr)} / validation {len(itv)} | meilleure époque {best_ep}")
    print(f"validation : AUC {met['auc']:.4f} | AUC par patient {met['auc_pw']:.4f} | sensibilité/spécificité par patient à 0,5 : "
          f"{met['sens_pw']:.3f}/{met['spec_pw']:.3f} | durée {round(time.time() - t0)} s")
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test (numpy, sans torch)
def selftest():
    import tempfile, shutil
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    mirror = {"FP1": "FP2", "F7": "F8", "T3": "T4", "T5": "T6", "O1": "O2", "F3": "F4", "C3": "C4", "P3": "P4"}
    mirror.update({v: k for k, v in list(mirror.items())})       # le miroir est symétrique : gauche <-> droite
    ok_mirror = all(DERIVATIONS[FLIP_PERM[i]] == (mirror[DERIVATIONS[i][0]], mirror[DERIVATIONS[i][1]]) for i in range(16))
    check("FLIP_PERM : miroir exact des 16 dérivations latérales", ok_mirror)
    check("FLIP_PERM : involution, FZ-CZ et CZ-PZ inchangés", [FLIP_PERM[FLIP_PERM[i]] for i in range(18)] == list(range(18)) and FLIP_PERM[16:] == [16, 17])
    # cache synthétique
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "cache", "train"))
        rows, sp = [], []
        valid = np.ones(18, bool)
        for i in range(12):
            pat = f"p{i // 2:02d}"
            n = 30 if i == 0 else 8                       # le patient p00 est très long
            x = (rng.standard_normal((n, 18, WIN)) * 15).astype(np.float16)
            x[:2] = 0                                      # 2 fenêtres plates
            y = np.array([1] * (n // 2) + [0] * (n - n // 2), np.int8)
            np.savez(os.path.join(tmp, "cache", "train", f"{pat}_s001_t00{i % 2}.npz"), x=x, start_sample=np.arange(n) * 500, m1=y,
                     m2=np.full(n, -1, np.int8), eid=np.full(n, -1, np.int16), valid=valid)
            rows.append(dict(split="train", recording=f"{pat}_s001_t00{i % 2}", patient=pat, status="ok"))
        for pat in sorted({r["patient"] for r in rows}):
            sp.append(dict(patient=pat, role="val" if pat in ("p04", "p05") else "train"))
        pd.DataFrame(rows).to_csv(os.path.join(tmp, "cache", "manifest.csv"), index=False)
        pd.DataFrame(sp).to_csv(os.path.join(tmp, "split.csv"), index=False)
        D = load_cache(os.path.join(tmp, "cache"), os.path.join(tmp, "split.csv"), verbose=False)
        n_expected = sum((30 if i == 0 else 8) - 2 for i in range(12))
        check(f"load_cache : fenêtres plates retirées ({len(D['y'])} = {n_expected})", len(D["y"]) == n_expected)
        check("load_cache : rôles assignés par patient (aucun patient dans les deux rôles)",
              not (set(D["pid"][D["role"] == "train"]) & set(D["pid"][D["role"] == "val"])) and set(D["role"]) == {"train", "val"})
        itr = np.where(D["role"] == "train")[0]
        s = BalancedSampler(D["y"][itr], D["pid"][itr], seed=1)
        d = s.draw(20000)
        check(f"BalancedSampler : classes ~50/50 ({np.mean(D['y'][itr][d]):.3f})", abs(np.mean(D["y"][itr][d]) - 0.5) < 0.02)
        share_raw = float((D["pid"][itr] == D["pid"][itr][D["patients"][D['pid'][itr]] == 'p00'][0]).mean())
        share_drawn = float((D["pid"][itr][d] == D["pid"][itr][D["patients"][D['pid'][itr]] == 'p00'][0]).mean())
        check(f"BalancedSampler : le patient long pèse moins qu'à l'état brut ({share_drawn:.3f} < {share_raw:.3f})", share_drawn < share_raw)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # métriques pondérées par patient
    y = np.array([1] * 100 + [0] * 100 + [1] * 5 + [0] * 5)
    pid = np.array([0] * 200 + [1] * 10)
    p = np.array([0.9] * 100 + [0.1] * 100 + [0.1] * 5 + [0.9] * 5)     # patient 0 parfait (gros), patient 1 inversé (petit)
    m = val_metrics(y, p, pid)
    check(f"val_metrics : AUC simple élevée ({m['auc']:.3f}) mais AUC par patient plus basse ({m['auc_pw']:.3f})", m["auc"] > 0.9 and m["auc_pw"] < m["auc"])
    m2 = val_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 1, 0, 1]))
    check("val_metrics : classement parfait -> AUC 1, sensibilité/spécificité 1", m2["auc"] == 1.0 and m2["auc_pw"] == 1.0 and m2["bacc_pw"] == 1.0)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    print("NB : la partie torch (modèle, boucle d'entraînement) n'est PAS couverte ici : lancer --smoke dans Colab.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--out_dir", default=base + "m1_seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps_per_epoch", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit_per_role", type=int, default=None, help="n'utiliser que N enregistrements par rôle (essais)")
    ap.add_argument("--no_flip", dest="flip", action="store_false")
    ap.add_argument("--no_attn", action="store_true")
    ap.add_argument("--norm", choices=["window", "windowglobal", "const"], default="window",
                    help="window : division par l'écart-type de chaque canal (défaut) ; const : division par 30 µV (garde l'amplitude)")
    ap.add_argument("--hardneg_dir", default=None, help="dossier des fenêtres normales difficiles (étape 8) ajoutées à l'entraînement (rôle train)")
    ap.add_argument("--smoke", action="store_true", help="essai rapide : 60 enregistrements par rôle, 2 époques de 20 pas")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.smoke:
        a.limit_per_role, a.epochs, a.steps_per_epoch, a.batch = 60, 2, 20, 16
        a.out_dir = a.out_dir.rstrip("/") + "_smoke"
    train(a)
