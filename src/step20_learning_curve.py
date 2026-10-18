#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 20 -- Manque-t-on de DONNÉES pour le bilatéral ?  Courbe d'apprentissage en validation croisée par patient.  AUCUN accès à eval ni à Siena.

Principe. Mêmes 5 plis par patient que l'étape 7. Dans chaque pli, on n'entraîne B_asym80 (régression logistique, miroir, poids égaux par patient et classe)
qu'avec une FRACTION des patients d'entraînement (10, 25, 50, 75, 100 %), tirée au hasard (3 tirages pour les fractions < 100 %, graines 0, 1, 2) ; le test est toujours
le même pli complet. On rapporte l'exactitude équilibrée PAR ÉVÉNEMENT et les rappels G/D/B en fonction du nombre de patients d'entraînement.
Lecture décidée à l'avance : si la courbe est PLATE sur les derniers points (gain de 50 % à 100 % < 0,01 d'exactitude équilibrée et < 0,02 de rappel bilatéral),
des données supplémentaires DU MÊME TYPE n'amélioreraient pas ce modèle ; si elle monte encore, le manque de données est une explication plausible.
Une courbe d'apprentissage ne dit rien d'un autre type de données (labels cliniques, autres centres).

Usage :
  python step20_learning_curve.py [--limit_per_role N]
  python step20_learning_curve.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step7_features_m2_cv import patient_folds, fit_model, predict_model, N_FOLDS
from step4_train_m2 import m2_indices, event_aggregate, m2_metrics

FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
REPEATS = (0, 1, 2)
MODEL = "B_asym80"


def subsample_patients(pids_train, frac, seed):
    pats = np.array(sorted(set(pids_train)))
    if frac >= 1.0:
        return pats
    n = max(3, int(round(frac * len(pats))))
    return np.random.default_rng(seed).choice(pats, n, replace=False)


def curve(bp, y, pid, rec, eid, idx, fractions=FRACTIONS, repeats=REPEATS, n_folds=N_FOLDS):
    fold_of = patient_folds(pid[idx], n_folds)
    fold = np.array([fold_of[p] for p in pid[idx]])
    rows = []
    for frac in fractions:
        for seed in (repeats if frac < 1.0 else (0,)):
            P = np.zeros((len(idx), 3))
            n_tr = []
            for k in range(n_folds):
                tr_all, te = idx[fold != k], idx[fold == k]
                if len(te) == 0:
                    continue
                keep = subsample_patients(pid[tr_all], frac, seed * 100 + k)
                tr = tr_all[np.isin(pid[tr_all], keep)]
                n_tr.append(len(keep))
                P[fold == k] = predict_model(fit_model(MODEL, bp, y, pid, tr), bp[te])
            ev = event_aggregate(rec[idx], eid[idx], pid[idx], y[idx], P)
            m = m2_metrics(ev.y.values, ev.pred.values, ev.pid.values)
            rows.append(dict(frac=frac, seed=seed, n_patients_train=float(np.mean(n_tr)), bacc=m["bacc"], rG=m["recall"][0], rD=m["recall"][1], rB=m["recall"][2]))
    return pd.DataFrame(rows)


def report(df):
    g = df.groupby("frac").agg(n=("n_patients_train", "mean"), bacc=("bacc", "mean"), bacc_sd=("bacc", "std"), rG=("rG", "mean"), rD=("rD", "mean"),
                              rB=("rB", "mean"), rB_sd=("rB", "std"), tirages=("seed", "size")).reset_index()
    L = ["Courbe d'apprentissage, B_asym80, exactitude équilibrée par événement (moyenne sur les tirages ; écart-type entre tirages) :",
         "  fraction | patients d'entraînement (moy./pli) | tirages | équilibrée      | rappels G/D/B"]
    for _, r in g.iterrows():
        sd = "-" if np.isnan(r.bacc_sd) else f"±{r.bacc_sd:.3f}"
        L.append(f"  {r.frac:>7.0%} | {r.n:>34.0f} | {int(r.tirages):>7} | {r.bacc:.3f} {sd:<7} | {r.rG:.2f}/{r.rD:.2f}/{r.rB:.2f}")
    a, b = g[g.frac == 0.5].iloc[0], g[g.frac == 1.0].iloc[0]
    db, dr = b.bacc - a.bacc, b.rB - a.rB
    flat = db < 0.01 and dr < 0.02
    L.append(f"gain de 50 % à 100 % des patients : exactitude équilibrée {db:+.3f}, rappel bilatéral {dr:+.3f} -> "
             + ("courbe PLATE : plus de données du même type n'amélioreraient pas ce modèle" if flat else "la courbe MONTE encore : le manque de données est une explication plausible"))
    return L, g, flat


def main(a):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    D = load_cache(a.cache_dir, a.split_csv, roles=("train", "val"), limit_per_role=a.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx.sort()
    df = curve(bp, D["m2"], D["pid"], D["rec"], D["eid"], idx)
    os.makedirs(a.out_dir, exist_ok=True)
    df.to_csv(os.path.join(a.out_dir, "courbe.csv"), index=False)
    L, _, _ = report(df)
    print("<<<COPY_FROM_HERE>>>")
    print(f"{len(idx)} fenêtres M2, {len(set(D['pid'][idx]))} patients (train + val), {N_FOLDS} plis par patient")
    print("\n".join(L))
    print("<<<COPY_TO_HERE>>>")


def selftest():
    from step3b_baselines_m1 import features_all
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    s = subsample_patients(np.repeat(np.arange(40), 3), 0.25, 0)
    check(f"fraction 25 % de 40 patients = 10 patients distincts ({len(s)})", len(s) == 10 and len(set(s)) == 10)
    check("fraction 100 % = tous les patients", len(subsample_patients(np.repeat(np.arange(40), 3), 1.0, 0)) == 40)
    check("tirages reproductibles", np.array_equal(subsample_patients(np.arange(40), 0.5, 7), subsample_patients(np.arange(40), 0.5, 7)))
    rng = np.random.default_rng(0)
    n_pat, nrec_pp, per_rec = 40, 3, 4
    n_rec = n_pat * nrec_pp
    n = n_rec * per_rec
    recs = np.repeat(np.arange(n_rec), per_rec)
    pat_of_rec = np.arange(n_rec) // nrec_pp
    cls = (np.arange(n_rec) % 3)[recs]
    x = rng.standard_normal((n, 18, 1000)).astype(np.float32) * 10
    for c, cols in ((0, range(0, 8)), (1, range(8, 16)), (2, range(0, 16))):
        x[np.ix_(cls == c, list(cols))] *= 3
    role = np.where(pat_of_rec[recs] < 28, "train", "val")
    D = dict(x=x.astype(np.float16), m2=cls.astype(np.int64), y=np.ones(n, np.int64), pid=pat_of_rec[recs], rec=recs, eid=np.zeros(n, np.int64),
             valid=np.tile(np.ones(18, bool), (n, 1)), role=role, rec_names=np.array([f"rec{i:03d}" for i in range(n_rec)]),
             patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.tile(np.arange(per_rec) * 500, n_rec))
    _, _, bp, _ = features_all(D)
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx.sort()
    df = curve(bp, D["m2"], D["pid"], D["rec"], D["eid"], idx, repeats=(0, 1))
    check(f"{len(df)} lignes = 4 fractions x 2 tirages + 1 pour 100 %", len(df) == 9)
    L, g, flat = report(df)
    check("rapport : 5 fractions et une conclusion", len(g) == 5 and any("gain de 50 %" in l for l in L))
    check(f"jeu facile : courbe plate (la tâche est apprise dès peu de patients) -> {flat}", flat)
    check("le nombre de patients d'entraînement augmente avec la fraction", list(g.n) == sorted(g.n))
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--out_dir", default=base + "learning_curve")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
