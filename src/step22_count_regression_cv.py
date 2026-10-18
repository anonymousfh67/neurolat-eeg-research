#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 22 -- Prédire les COMPTES de canaux actifs (gauche, droite) puis appliquer la règle du label.  Validation croisée sur train + val SEULEMENT (ni eval, ni Siena).

Le label gauche / droite / bilatéral vient d'une règle sur deux nombres : nL et nR = canaux distincts annotés à gauche / à droite dans les 2 premières secondes de l'événement
(events.csv de l'étape 1, colonnes nL_channels et nR_channels). Les modèles précédents n'apprenaient que la classe finale et perdaient ces deux nombres, qui sont une supervision
plus riche (une crise à 2+2 canaux et une crise à 6+6 canaux sont toutes deux « bilatérales » mais pas également nettes).
Ici, M2 prédit log(1 + nL) et log(1 + nR) par fenêtre ; les prédictions sont moyennées par événement ; la règle est appliquée aux comptes prédits :
   bilatéral si nL_prédit >= t et nR_prédit >= t ; sinon le côté le plus grand.      (t = 1,5 : arrondi à >= 2, comme k = 2 de la règle ; décidé à l'avance)
Variantes décidées À L'AVANCE, toutes rapportées :
   V0  : B_asym80 (classification directe, étape 7), référence
   R1  : régression ridge (alpha = 10) sur les variables de B_asym80, t = 1,5
   R2  : R1 avec t choisi par validation croisée INTERNE (4 plis par patient, grille 1,0 ; 1,25 ; 1,5 ; 1,75 ; 2,0) -> aucun usage du pli externe
   R3  : arbres de gradient (profondeur 3, 150 itérations) pour log(1 + nL) et log(1 + nR), t = 1,5
Mêmes 5 plis par patient que l'étape 7, exactitude équilibrée PAR ÉVÉNEMENT, IC95 par bootstrap de patients, différences APPARIÉES avec V0. 3 comparaisons : exploratoire.

Usage :
  python step22_count_regression_cv.py --events_csv .../etape1_train_dev/events.csv [--cv_dir .../lr_cv]
  python step22_count_regression_cv.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step7_features_m2_cv import patient_folds, feat_B, mirror_bp, sample_weights, N_FOLDS
from step3_train_m1 import FLIP_PERM
from step4_train_m2 import m2_indices, FLIP_LABEL
from step11_label_clarity import confusion_by_patient, boot_idx, boot_bacc, bacc_acc, ci, fmt

T_DEFAULT = 1.5
T_GRID = (1.0, 1.25, 1.5, 1.75, 2.0)
INNER = 4
KEY = ["recording", "eid"]


def decide(nL, nR, t):
    """Règle appliquée à des comptes (prédits) : 0 gauche, 1 droite, 2 bilatéral."""
    nL, nR = np.asarray(nL, float), np.asarray(nR, float)
    out = np.where(nL >= nR, 0, 1)
    return np.where((nL >= t) & (nR >= t), 2, out)


def window_counts(D, events_csv):
    t = pd.read_csv(events_csv)
    m = {(r, int(e)): (int(l), int(rr)) for r, e, l, rr in zip(t.recording, t.event_idx, t.nL_channels, t.nR_channels)}
    names = D["rec_names"]
    tL = np.full(len(D["m2"]), np.nan)
    tR = np.full(len(D["m2"]), np.nan)
    for i, (r, e) in enumerate(zip(D["rec"], D["eid"])):
        v = m.get((names[r], int(e)))
        if v is not None and e >= 0:
            tL[i], tR[i] = v
    return tL, tR


def fit_reg(kind, bp, tL, tR, y, pid, tr):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.concatenate([feat_B(bp[tr]), feat_B(mirror_bp(bp[tr]))])
    YL = np.concatenate([np.log1p(tL[tr]), np.log1p(tR[tr])])           # miroir : gauche <-> droite
    YR = np.concatenate([np.log1p(tR[tr]), np.log1p(tL[tr])])
    w = sample_weights(np.concatenate([y[tr], FLIP_LABEL[y[tr]]]), np.concatenate([pid[tr], pid[tr]]))
    if kind == "ridge":
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        return kind, sc, [Ridge(alpha=10.0).fit(Xs, Y, sample_weight=w) for Y in (YL, YR)]
    return kind, None, [HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=0).fit(X, Y, sample_weight=w)
                        for Y in (YL, YR)]


def predict_reg(model, bp):
    kind, sc, ms = model
    X = feat_B(bp)
    if sc is not None:
        X = sc.transform(X)
    return np.stack([m.predict(X) for m in ms], axis=1)                  # (n, 2) : log(1 + nL), log(1 + nR)


def event_counts(rec, eid, pid, y, Z):
    df = pd.DataFrame(dict(rec=rec, eid=eid, pid=pid, y=y, zL=Z[:, 0], zR=Z[:, 1]))
    g = df.groupby(["rec", "eid"]).agg(y=("y", "first"), pid=("pid", "first"), zL=("zL", "mean"), zR=("zR", "mean")).reset_index()
    g["nL"], g["nR"] = np.expm1(g.zL), np.expm1(g.zR)
    return g


def bacc_event(g, t):
    pred = decide(g.nL, g.nR, t)
    r = [np.mean(pred[g.y.values == c] == c) for c in range(3) if (g.y.values == c).any()]
    return float(np.mean(r))


def choose_t(g):
    return max(((bacc_event(g, t), -abs(t - T_DEFAULT), t) for t in T_GRID))[2]


def cv_reg(kind, bp, tL, tR, y, pid, rec, eid, idx, nested=False):
    fold_of = patient_folds(pid[idx], N_FOLDS)
    fold = np.array([fold_of[p] for p in pid[idx]])
    Z = np.zeros((len(idx), 2))
    t_of = {}
    for k in range(N_FOLDS):
        tr, te = idx[fold != k], idx[fold == k]
        if len(te) == 0:
            continue
        if nested:
            f2 = patient_folds(pid[tr], INNER, seed=1)
            f2w = np.array([f2[p] for p in pid[tr]])
            Zin = np.zeros((len(tr), 2))
            for j in range(INNER):
                a, b = tr[f2w != j], tr[f2w == j]
                if len(b):
                    Zin[f2w == j] = predict_reg(fit_reg(kind, bp, tL, tR, y, pid, a), bp[b])
            t_of[k] = choose_t(event_counts(rec[tr], eid[tr], pid[tr], y[tr], Zin))
        Z[fold == k] = predict_reg(fit_reg(kind, bp, tL, tR, y, pid, tr), bp[te])
    return Z, fold, t_of


def analyse(D, bp, tL, tR, ref, B=2000):
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx = np.sort(idx[~np.isnan(tL[idx])])
    y, pid, rec, eid = D["m2"], D["pid"], D["rec"], D["eid"]
    evs = {}
    Zr, fold, t_of = cv_reg("ridge", bp, tL, tR, y, pid, rec, eid, idx, nested=True)
    g = event_counts(rec[idx], eid[idx], pid[idx], y[idx], Zr)
    f_of_pid = dict(zip(pid[idx], fold))
    g1 = g.copy(); g1["pred"] = decide(g.nL, g.nR, T_DEFAULT); evs["R1 ridge, t = 1,5"] = g1
    g2 = g.copy(); tt = np.array([t_of[f_of_pid[p]] for p in g.pid]); g2["pred"] = np.where((g.nL >= tt) & (g.nR >= tt), 2, np.where(g.nL >= g.nR, 0, 1))
    evs["R2 ridge, t choisi en interne"] = g2
    Zh, _, _ = cv_reg("hgb", bp, tL, tR, y, pid, rec, eid, idx)
    g3 = event_counts(rec[idx], eid[idx], pid[idx], y[idx], Zh); g3["pred"] = decide(g3.nL, g3.nR, T_DEFAULT); evs["R3 arbres, t = 1,5"] = g3
    for k in evs:
        evs[k]["recording"] = D["rec_names"][evs[k].rec.values]
    base = evs["R1 ridge, t = 1,5"][KEY + ["pid", "y"]].merge(ref, on=KEY, how="left")
    n_lost = int(base.pred_ref.isna().sum())
    evs = {"V0 B_asym80 (classification directe)": base[base.pred_ref.notna()].rename(columns={"pred_ref": "pred"}), **evs}
    common = set.intersection(*[set(map(tuple, e[KEY].values)) for e in evs.values()])
    for k in evs:
        e = evs[k]
        evs[k] = e[[tuple(x) in common for x in e[KEY].values]].sort_values(KEY).reset_index(drop=True)
    keys = list(evs)
    patients = sorted(set(evs[keys[0]].pid))
    bi = boot_idx(len(patients), B)
    C = {k: confusion_by_patient(evs[k].pid, evs[k].y, evs[k].pred, patients) for k in keys}
    L = [f"{len(idx)} fenêtres, {len(patients)} patients, {len(evs[keys[0]])} événements (réf. sans correspondance : {n_lost}) ; t choisis par pli (R2) : " + ", ".join(f"{t_of[k]:g}" for k in sorted(t_of)),
         "Exactitude équilibrée par événement :"]
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


def ref_events(cv_dir):
    p = os.path.join(cv_dir, "B_asym80_val_scores.csv")
    if not os.path.exists(p):
        raise SystemExit(f"ERREUR : {p} absent.")
    d = pd.read_csv(p)
    g = d.groupby(KEY).agg(pL=("pL", "mean"), pR=("pR", "mean"), pB=("pB", "mean")).reset_index()
    g["pred_ref"] = g[["pL", "pR", "pB"]].values.argmax(1)
    return g[KEY + ["pred_ref"]]


def main(a):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    D = load_cache(a.cache_dir, a.split_csv, roles=("train", "val"), limit_per_role=a.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    tL, tR = window_counts(D, a.events_csv)
    L, evs = analyse(D, bp, tL, tR, ref_events(a.cv_dir))
    os.makedirs(a.out_dir, exist_ok=True)
    for k, e in evs.items():
        e.to_csv(os.path.join(a.out_dir, k.split()[0] + "_events.csv"), index=False)
    print("<<<COPY_FROM_HERE>>>")
    print("\n".join(L))
    print("<<<COPY_TO_HERE>>>")


def selftest():
    import tempfile, shutil
    from step3b_baselines_m1 import features_all
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    check("règle sur comptes : (3, 0) -> gauche, (0, 4) -> droite, (2, 2) -> bilatéral, (1, 1) -> non bilatéral, (1,6 ; 1,6) -> bilatéral",
          list(decide([3, 0, 2, 1, 1.6], [0, 4, 2, 1, 1.6], 1.5)) == [0, 1, 2, 0, 2])
    rng = np.random.default_rng(0)
    n_pat, nrec_pp, per_rec = 40, 3, 4
    n_rec = n_pat * nrec_pp
    n = n_rec * per_rec
    recs = np.repeat(np.arange(n_rec), per_rec)
    pat_of_rec = np.arange(n_rec) // nrec_pp
    cls_rec = np.arange(n_rec) % 3
    cls = cls_rec[recs]
    nl_rec = np.array([4, 0, 3])[cls_rec]
    nr_rec = np.array([0, 4, 3])[cls_rec]
    x = rng.standard_normal((n, 18, 1000)).astype(np.float32) * 10
    for c, cols in ((0, range(0, 8)), (1, range(8, 16)), (2, range(0, 16))):
        x[np.ix_(cls == c, list(cols))] *= 3
    role = np.where(pat_of_rec[recs] < 28, "train", "val")
    D = dict(x=x.astype(np.float16), m2=cls.astype(np.int64), y=np.ones(n, np.int64), pid=pat_of_rec[recs], rec=recs, eid=np.zeros(n, np.int64),
             valid=np.tile(np.ones(18, bool), (n, 1)), role=role, rec_names=np.array([f"rec{i:03d}" for i in range(n_rec)]),
             patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.tile(np.arange(per_rec) * 500, n_rec))
    tmp = tempfile.mkdtemp()
    try:
        ev = os.path.join(tmp, "events.csv")
        pd.DataFrame(dict(recording=D["rec_names"], event_idx=0, nL_channels=nl_rec, nR_channels=nr_rec)).to_csv(ev, index=False)
        tL, tR = window_counts(D, ev)
        check("comptes par fenêtre lus depuis events.csv", np.array_equal(tL[::per_rec], nl_rec) and np.array_equal(tR[::per_rec], nr_rec))
        _, _, bp, _ = features_all(D)
        e_ref = pd.DataFrame(dict(recording=D["rec_names"], eid=0, pred_ref=cls_rec))
        L, evs = analyse(D, bp, tL, tR, e_ref, B=200)
        txt = "\n".join(L)
        check(f"4 lignes (V0, R1, R2, R3) et mêmes événements ({ {len(e) for e in evs.values()} } = {{{n_rec}}})", len(evs) == 4 and {len(e) for e in evs.values()} == {n_rec})
        check("rapport : 3 différences appariées et les t choisis", txt.count("différence appariée") == 3 and "t choisis par pli" in txt)
        r1 = [e for k, e in evs.items() if k.startswith("R1")][0]
        acc = float((r1.y == r1.pred).mean())
        check(f"jeu facile : la règle sur comptes prédits retrouve les classes ({acc:.2f} > 0,9)", acc > 0.9)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--events_csv", default=base + "etape1_train_dev/events.csv")
    ap.add_argument("--cv_dir", default=base + "lr_cv")
    ap.add_argument("--out_dir", default=base + "count_cv")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
