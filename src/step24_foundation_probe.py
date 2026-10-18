#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 24 -- Caractéristiques d'un modèle de fondation EEG pré-entraîné (CBraMod) + sonde linéaire.  Validation croisée sur train + val SEULEMENT (ni eval, ni Siena).

Tous les essais précédents partent de rien ou de variables calculées à la main. Ici on réutilise un encodeur Transformer pré-entraîné par auto-supervision sur un très grand corpus
EEG (CBraMod, ICLR 2025, code https://github.com/wjq-learning/CBraMod, poids https://huggingface.co/weighting666/CBraMod ; pré-entraîné sur des enregistrements TUEG).
Principe : chaque fenêtre (18 dérivations bipolaires, 4 s, 250 Hz) est rééchantillonnée à 200 Hz (facteur 4/5), divisée par 100 (µV/100, comme dans le code de CBraMod),
découpée en 4 patchs de 200 points, passée dans l'encodeur GELÉ ; la représentation de chaque canal est la moyenne de ses 4 patchs (200 dimensions). On en tire 600 variables :
   d = moyenne(hémisphère gauche) - moyenne(hémisphère droit) ; m = moyenne des deux ; mid = moyenne des 2 dérivations médianes (200 dimensions chacune).
Le miroir gauche/droite d'entraînement se fait sur l'ENTRÉE (échange des deux chaînes de canaux, puis nouvelle extraction), pas sur les variables.
Limites annoncées : (1) la sonde est LINÉAIRE sur un encodeur gelé ; un réglage fin (« fine-tuning ») pourrait faire mieux ou pire ; (2) l'encodeur a été pré-entraîné sur des
enregistrements TUEG, dont TUSZ est un sous-ensemble : il a pu voir, SANS étiquette, des enregistrements de patients d'eval. Aucun label n'est en jeu, mais toute utilisation sur
eval ou sur le papier devra le déclarer ; ici, seule la validation croisée sur train + val est utilisée ; (3) le pré-entraînement utilisait des canaux référentiels, ici on donne des
dérivations bipolaires, comme dans l'évaluation TUEV de CBraMod.
Variantes décidées À L'AVANCE : V0 = B_asym80 (référence) ; P1 = régression logistique (C = 0,1) sur les 600 variables CBraMod seules ; P2 = idem sur B_asym80 + 600 variables.
Mêmes 5 plis par patient que l'étape 7, exactitude équilibrée par événement, IC95 par bootstrap de patients, différences appariées avec V0. 2 comparaisons : exploratoire.

Usage (Colab, GPU) :
  !git clone --depth 1 https://github.com/wjq-learning/CBraMod /content/CBraMod ; !pip -q install einops
  python step24_foundation_probe.py extract --smoke          (64 fenêtres : vérifie formes et temps)
  python step24_foundation_probe.py extract                  (toutes les fenêtres train + val, écrit foundation/feats.npz)
  python step24_foundation_probe.py probe
  python step24_foundation_probe.py --selftest
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step7_features_m2_cv import patient_folds, feat_B, mirror_bp, sample_weights, N_FOLDS
from step4_train_m2 import m2_indices, event_aggregate, FLIP_LABEL
from step11_label_clarity import confusion_by_patient, boot_idx, boot_bacc, bacc_acc, ci, fmt

KEY = ["recording", "eid"]
NPOW = 90
D_EMB = 200
# échange des deux chaînes de canaux (0-7 <-> 8-15), médianes inchangées : même opération que le miroir de l'étape 3
MIRROR_CH = np.array(list(range(8, 16)) + list(range(0, 8)) + [16, 17])


def pool_embeddings(E):
    """E : (n, 18, P, 200) -> (n, 600) = [gauche - droite, (gauche + droite)/2, médianes], chaque bloc de 200, moyenne des patchs."""
    ch = E.mean(axis=2)                                           # (n, 18, 200)
    L, R, M = ch[:, :8].mean(axis=1), ch[:, 8:16].mean(axis=1), ch[:, 16:18].mean(axis=1)
    return np.concatenate([L - R, (L + R) / 2, M], axis=1).astype(np.float32)


def prepare_input(x):
    """x : (b, 18, 1000) µV à 250 Hz -> (b, 18, 4, 200) à 200 Hz, en unités µV / 100."""
    from scipy.signal import resample_poly
    y = resample_poly(np.asarray(x, np.float32), 4, 5, axis=-1)           # 1000 -> 800 points
    return (y / 100.0).reshape(len(y), y.shape[1], 4, 200)


def extract(a):
    import torch
    sys.path.insert(0, a.cbramod_dir)
    from models.cbramod import CBraMod
    from step3_train_m1 import load_cache
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = a.weights
    if not os.path.exists(w):
        try:
            from huggingface_hub import hf_hub_download
            w = hf_hub_download(repo_id="weighting666/CBraMod", filename="pretrained_weights.pth")
        except Exception as e:
            raise SystemExit(f"ERREUR : poids introuvables ({w}) et téléchargement impossible ({e}). Télécharger pretrained_weights.pth depuis https://huggingface.co/weighting666/CBraMod puis relancer avec --weights.")
    model = CBraMod().to(device)
    model.load_state_dict(torch.load(w, map_location=device))
    model.proj_out = torch.nn.Identity()
    model.eval()
    D = load_cache(a.cache_dir, a.split_csv, roles=("train", "val"), limit_per_role=a.limit_per_role, verbose=False)
    idx = np.sort(np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")]))
    if a.smoke:
        idx = idx[:64]
    F = np.zeros((len(idx), 3 * D_EMB), np.float16)
    Fm = np.zeros_like(F)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(idx), a.batch):
            xb = np.asarray(D["x"][idx[i:i + a.batch]], np.float32)
            for out, xin in ((F, xb), (Fm, xb[:, MIRROR_CH])):
                E = model(torch.from_numpy(prepare_input(xin)).to(device)).float().cpu().numpy()
                out[i:i + len(xb)] = pool_embeddings(E)
            if (i // a.batch) % 50 == 0:
                print(f"  {i + len(xb)}/{len(idx)} fenêtres ({time.time() - t0:.0f} s)", flush=True)
    print(f"formes : encodeur -> {E.shape}, variables {F.shape} ; {time.time() - t0:.0f} s ; NaN : {int(np.isnan(F.astype(np.float32)).sum())}", flush=True)
    if not a.smoke:
        os.makedirs(os.path.dirname(a.feats), exist_ok=True)
        np.savez(a.feats, idx=idx, F=F, Fm=Fm)
        print(f"écrit {a.feats}")


# ------------------------------------------------------------------ sonde
def fit_probe(mode, bp, F, Fm, y, pid, tr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    def feats(bpx, Fx):
        return Fx if mode == "probe_only" else np.concatenate([feat_B(bpx), Fx], axis=1)
    X = np.concatenate([feats(bp[tr], F[tr]), feats(mirror_bp(bp[tr]), Fm[tr])])
    yy = np.concatenate([y[tr], FLIP_LABEL[y[tr]]])
    w = sample_weights(yy, np.concatenate([pid[tr], pid[tr]]))
    sc = StandardScaler().fit(X)
    return mode, sc, LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(X), yy, sample_weight=w), feats


def predict_probe(model, bp, F):
    mode, sc, clf, feats = model
    P = np.zeros((len(bp), 3))
    P[:, clf.classes_] = clf.predict_proba(sc.transform(feats(bp, F)))
    return P


def cv_probe(mode, bp, F, Fm, y, pid, idx):
    fold_of = patient_folds(pid[idx], N_FOLDS)
    fold = np.array([fold_of[p] for p in pid[idx]])
    P = np.zeros((len(idx), 3))
    for k in range(N_FOLDS):
        tr, te = idx[fold != k], idx[fold == k]
        if len(te):
            P[fold == k] = predict_probe(fit_probe(mode, bp, F, Fm, y, pid, tr), bp[te], F[te])
    return P


def ref_events(cv_dir):
    p = os.path.join(cv_dir, "B_asym80_val_scores.csv")
    if not os.path.exists(p):
        raise SystemExit(f"ERREUR : {p} absent.")
    d = pd.read_csv(p)
    g = d.groupby(KEY).agg(pL=("pL", "mean"), pR=("pR", "mean"), pB=("pB", "mean")).reset_index()
    g["pred_ref"] = g[["pL", "pR", "pB"]].values.argmax(1)
    return g[KEY + ["pred_ref"]]


def analyse(D, bp, Fall, Fmall, idx, ref, B=2000):
    y, pid, rec, eid = D["m2"], D["pid"], D["rec"], D["eid"]
    evs = {}
    for tag, mode in (("P1 sonde CBraMod seule", "probe_only"), ("P2 B_asym80 + CBraMod", "both")):
        P = cv_probe(mode, bp, Fall, Fmall, y, pid, idx)
        e = event_aggregate(rec[idx], eid[idx], pid[idx], y[idx], P)
        e["recording"] = D["rec_names"][e.rec.values]
        evs[tag] = e
    first = next(iter(evs.values()))
    base = first[KEY + ["pid", "y"]].merge(ref, on=KEY, how="left")
    n_lost = int(base.pred_ref.isna().sum())
    evs = {"V0 B_asym80": base[base.pred_ref.notna()].rename(columns={"pred_ref": "pred"}), **evs}
    common = set.intersection(*[set(map(tuple, e[KEY].values)) for e in evs.values()])
    for k in evs:
        e = evs[k]
        evs[k] = e[[tuple(x) in common for x in e[KEY].values]].sort_values(KEY).reset_index(drop=True)
    keys = list(evs)
    patients = sorted(set(evs[keys[0]].pid))
    bi = boot_idx(len(patients), B)
    C = {k: confusion_by_patient(evs[k].pid, evs[k].y, evs[k].pred, patients) for k in keys}
    L = [f"{len(idx)} fenêtres, {len(patients)} patients, {len(evs[keys[0]])} événements (réf. sans correspondance : {n_lost})", "Exactitude équilibrée par événement :"]
    for k in keys:
        L.append(fmt(C[k], bi, k))
    b0 = boot_bacc(C[keys[0]], bi)
    for k in keys[1:]:
        d = boot_bacc(C[k], bi) - b0
        lo, hi = ci(d)
        dm = bacc_acc(C[k])[0] - bacc_acc(C[keys[0]])[0]
        v = "MEILLEUR (IC exclut 0)" if lo > 0 else ("moins bon (IC exclut 0)" if hi < 0 else "pas de différence détectable (IC contient 0)")
        L.append(f"différence appariée {k} - V0 : {dm:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}")
    return L, evs


def probe(a):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    z = np.load(a.feats)
    D = load_cache(a.cache_dir, a.split_csv, roles=("train", "val"), limit_per_role=a.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    idx = z["idx"]
    n = len(D["m2"])
    Fall = np.zeros((n, 3 * D_EMB), np.float32); Fall[idx] = z["F"]
    Fmall = np.zeros((n, 3 * D_EMB), np.float32); Fmall[idx] = z["Fm"]
    L, evs = analyse(D, np.asarray(bp, np.float32).reshape(n, -1), Fall, Fmall, idx, ref_events(a.cv_dir))
    os.makedirs(a.out_dir, exist_ok=True)
    for k, e in evs.items():
        e.to_csv(os.path.join(a.out_dir, k.split()[0] + "_events.csv"), index=False)
    print("<<<COPY_FROM_HERE>>>")
    print("\n".join(L))
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test (sans torch ni poids : encodeur factice)
def selftest():
    from step3b_baselines_m1 import features_all
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 18, 1000)).astype(np.float32) * 20
    u = prepare_input(x)
    check(f"entrée de l'encodeur : (n, 18, 4, 200) ({u.shape}) en unités µV/100", u.shape == (6, 18, 4, 200) and abs(u.std() - 0.2) < 0.05)
    check("échange des chaînes : involutif et laisse les médianes", np.array_equal(MIRROR_CH[MIRROR_CH], np.arange(18)) and list(MIRROR_CH[16:]) == [16, 17])
    E = rng.standard_normal((6, 18, 4, 200)).astype(np.float32)
    f = pool_embeddings(E)
    fm = pool_embeddings(E[:, MIRROR_CH])
    check("variables : 600 colonnes", f.shape == (6, 600))
    check("si l'encodeur est équivariant au miroir, gauche - droite change de signe et le reste est inchangé",
          np.allclose(fm[:, :200], -f[:, :200], atol=1e-5) and np.allclose(fm[:, 200:], f[:, 200:], atol=1e-5))
    # pipeline de sonde sur un jeu où l'« embedding » factice encode le bilatéral
    n_pat, nrec_pp, per_rec = 40, 3, 4
    n_rec = n_pat * nrec_pp
    N = n_rec * per_rec
    recs = np.repeat(np.arange(n_rec), per_rec)
    pat_of_rec = np.arange(n_rec) // nrec_pp
    cls = (np.arange(n_rec) % 3)[recs]
    X = rng.standard_normal((N, 18, 1000)).astype(np.float32) * 10
    for c, cols in ((0, range(0, 8)), (1, range(8, 16)), (2, range(0, 16))):
        X[np.ix_(cls == c, list(cols))] *= 3
    role = np.where(pat_of_rec[recs] < 28, "train", "val")
    D = dict(x=X.astype(np.float16), m2=cls.astype(np.int64), y=np.ones(N, np.int64), pid=pat_of_rec[recs], rec=recs, eid=np.zeros(N, np.int64),
             valid=np.tile(np.ones(18, bool), (N, 1)), role=role, rec_names=np.array([f"rec{i:03d}" for i in range(n_rec)]),
             patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.tile(np.arange(per_rec) * 500, n_rec))
    _, _, bp, _ = features_all(D)
    bp = np.asarray(bp, np.float32).reshape(N, -1)
    def fake_encoder(xx):          # « encodeur » factice équivariant : log-puissance par canal répétée sur 4 patchs x 200 dimensions
        lp = np.log(xx.std(axis=-1) + 1e-6)[..., None, None]
        return np.broadcast_to(lp, (len(xx), 18, 4, 200)) + 0.01 * rng.standard_normal((len(xx), 18, 4, 200))
    xs = np.asarray(D["x"], np.float32)
    F = pool_embeddings(fake_encoder(xs)); Fm = pool_embeddings(fake_encoder(xs[:, MIRROR_CH]))
    idx = np.sort(np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")]))
    e_ref = pd.DataFrame(dict(recording=D["rec_names"], eid=0, pred_ref=(np.arange(n_rec) % 3)))
    L, evs = analyse(D, bp, F, Fm, idx, e_ref, B=200)
    txt = "\n".join(L)
    check(f"3 lignes (V0, P1, P2) et mêmes événements ({ {len(e) for e in evs.values()} } = {{{n_rec}}})", len(evs) == 3 and {len(e) for e in evs.values()} == {n_rec})
    check("rapport : 2 différences appariées", txt.count("différence appariée") == 2)
    p1 = [e for k, e in evs.items() if k.startswith("P1")][0]
    acc = float((p1.y == p1.pred).mean())
    check(f"jeu facile : la sonde apprend ({acc:.2f} > 0,9)", acc > 0.9)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    base = "/content/drive/MyDrive/projet_propre/"
    for name in ("extract", "probe"):
        s = sub.add_parser(name)
        s.add_argument("--cache_dir", default=base + "cache")
        s.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
        s.add_argument("--feats", default=base + "foundation/feats.npz")
        s.add_argument("--cv_dir", default=base + "lr_cv")
        s.add_argument("--out_dir", default=base + "foundation")
        s.add_argument("--limit_per_role", type=int, default=None)
        if name == "extract":
            s.add_argument("--cbramod_dir", default="/content/CBraMod")
            s.add_argument("--weights", default="/content/CBraMod/pretrained_weights/pretrained_weights.pth")
            s.add_argument("--batch", type=int, default=128)
            s.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else probe(a)
