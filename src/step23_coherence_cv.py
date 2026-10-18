#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 23 -- Synchronisation INTER-HÉMISPHÉRIQUE (cohérence) pour reconnaître le bilatéral.  Validation croisée sur train + val SEULEMENT (ni eval, ni Siena).

Tous les essais précédents n'utilisaient que la PUISSANCE par bande (combien d'activité à gauche / à droite). Une crise bilatérale se distingue aussi par des hémisphères qui
s'activent ENSEMBLE, alors que dans une crise unilatérale ils restent indépendants. Ici, pour chaque fenêtre de 4 s, pour chacune des 8 paires miroir (dérivation gauche i
et dérivation droite i + 8) et chacune des 5 bandes (0,5-4 / 4-8 / 8-13 / 13-30 / 30-70 Hz) :
   - cohérence (module au carré) : synchronisation de phase ET d'amplitude
   - cohérence IMAGINAIRE (valeur absolue) : synchronisation avec décalage de phase, insensible à la conduction de volume (un même signal vu par deux électrodes voisines)
   estimées sur 7 segments de 1 s (recouvrement 50 %, fenêtre de Hann), moyennées dans la bande. 80 variables, invariantes au miroir.
Variantes décidées À L'AVANCE, toutes rapportées :
   V0 : B_asym80 (puissance seule, étape 7), référence
   C1 : régression logistique sur B_asym80 + cohérence (160 variables)
   C2 : arbres de gradient (profondeur 3, 150 itérations) sur B_asym80 + cohérence
   C3 : régression logistique sur la cohérence SEULE (diagnostic : l'information de synchronisation existe-t-elle sans la puissance ?)
Mêmes 5 plis par patient que l'étape 7, exactitude équilibrée par événement, IC95 par bootstrap de patients, différences APPARIÉES avec V0. 3 comparaisons : exploratoire.

Usage :
  python step23_coherence_cv.py [--cv_dir .../lr_cv] [--limit_per_role N]
  python step23_coherence_cv.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step7_features_m2_cv import patient_folds, feat_B, mirror_bp, sample_weights, N_FOLDS
from step4_train_m2 import m2_indices, event_aggregate, FLIP_LABEL
from step11_label_clarity import confusion_by_patient, boot_idx, boot_bacc, bacc_acc, ci, fmt

SEG, HOP = 250, 125
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 70))       # bornes en Hz (résolution 1 Hz pour des segments de 1 s à 250 Hz)
NPOW = 90
KEY = ["recording", "eid"]


def coherence_features(x, chunk=1000):
    """x : (n, 18, T>=1000) en µV -> (n, 80) : [cohérence, cohérence imaginaire] x 8 paires miroir x 5 bandes (ordre : type, paire, bande)."""
    n, _, T = x.shape
    starts = list(range(0, T - SEG + 1, HOP))
    win = np.hanning(SEG).astype(np.float32)
    out = np.zeros((n, 2, 8, len(BANDS)), np.float32)
    for c0 in range(0, n, chunk):
        xs = x[c0:c0 + chunk].astype(np.float32)
        segs = np.stack([xs[:, :, s:s + SEG] for s in starts], axis=2)            # (m, 18, S, SEG)
        segs = segs - segs.mean(axis=3, keepdims=True)
        F = np.fft.rfft(segs * win, axis=3)                                        # (m, 18, S, 126)
        A, Bc = F[:, :8], F[:, 8:16]
        Sxy = (A * np.conj(Bc)).mean(axis=2)                                       # (m, 8, 126)
        Sxx = (np.abs(A) ** 2).mean(axis=2)
        Syy = (np.abs(Bc) ** 2).mean(axis=2)
        den = Sxx * Syy + 1e-12
        coh = np.abs(Sxy) ** 2 / den
        icoh = np.abs(Sxy.imag) / np.sqrt(den)
        for b, (lo, hi) in enumerate(BANDS):
            out[c0:c0 + chunk, 0, :, b] = coh[:, :, lo:hi].mean(axis=2)
            out[c0:c0 + chunk, 1, :, b] = icoh[:, :, lo:hi].mean(axis=2)
    return out.reshape(n, -1)


def feat_ext(Z, mode):
    pw, co = Z[:, :NPOW], Z[:, NPOW:]
    if mode == "coh_only":
        return co
    return np.concatenate([feat_B(pw), co], axis=1)


def mirror_ext(Z):
    return np.concatenate([mirror_bp(Z[:, :NPOW]), Z[:, NPOW:]], axis=1)             # la cohérence entre paires miroir est invariante au miroir


def fit_ext(kind, mode, Z, y, pid, tr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.concatenate([feat_ext(Z[tr], mode), feat_ext(mirror_ext(Z[tr]), mode)])
    yy = np.concatenate([y[tr], FLIP_LABEL[y[tr]]])
    w = sample_weights(yy, np.concatenate([pid[tr], pid[tr]]))
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=0).fit(X, yy, sample_weight=w)
        return kind, mode, None, clf
    sc = StandardScaler().fit(X)
    return kind, mode, sc, LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), yy, sample_weight=w)


def predict_ext(model, Z):
    kind, mode, sc, clf = model
    X = feat_ext(Z, mode)
    P = np.zeros((len(Z), 3))
    P[:, clf.classes_] = clf.predict_proba(X if sc is None else sc.transform(X))
    return P


def cv_ext(kind, mode, Z, y, pid, idx):
    fold_of = patient_folds(pid[idx], N_FOLDS)
    fold = np.array([fold_of[p] for p in pid[idx]])
    P = np.zeros((len(idx), 3))
    for k in range(N_FOLDS):
        tr, te = idx[fold != k], idx[fold == k]
        if len(te):
            P[fold == k] = predict_ext(fit_ext(kind, mode, Z, y, pid, tr), Z[te])
    return P


def ref_events(cv_dir):
    p = os.path.join(cv_dir, "B_asym80_val_scores.csv")
    if not os.path.exists(p):
        raise SystemExit(f"ERREUR : {p} absent.")
    d = pd.read_csv(p)
    g = d.groupby(KEY).agg(pL=("pL", "mean"), pR=("pR", "mean"), pB=("pB", "mean")).reset_index()
    g["pred_ref"] = g[["pL", "pR", "pB"]].values.argmax(1)
    return g[KEY + ["pred_ref"]]


def analyse(D, Z, ref, B=2000):
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx.sort()
    y, pid, rec, eid = D["m2"], D["pid"], D["rec"], D["eid"]
    evs = {}
    for tag, kind, mode in (("C1 LR puissance + cohérence", "lr", "both"), ("C2 arbres puissance + cohérence", "hgb", "both"), ("C3 LR cohérence seule", "lr", "coh_only")):
        P = cv_ext(kind, mode, Z, y, pid, idx)
        e = event_aggregate(rec[idx], eid[idx], pid[idx], y[idx], P)
        e["recording"] = D["rec_names"][e.rec.values]
        evs[tag] = e
    first = next(iter(evs.values()))
    base = first[KEY + ["pid", "y"]].merge(ref, on=KEY, how="left")
    n_lost = int(base.pred_ref.isna().sum())
    evs = {"V0 B_asym80 (puissance seule)": base[base.pred_ref.notna()].rename(columns={"pred_ref": "pred"}), **evs}
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


def main(a):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    D = load_cache(a.cache_dir, a.split_csv, roles=("train", "val"), limit_per_role=a.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    print("cohérence inter-hémisphérique en cours de calcul...", flush=True)
    Z = np.concatenate([np.asarray(bp, np.float32).reshape(len(bp), -1), coherence_features(D["x"])], axis=1)
    L, evs = analyse(D, Z, ref_events(a.cv_dir))
    os.makedirs(a.out_dir, exist_ok=True)
    for k, e in evs.items():
        e.to_csv(os.path.join(a.out_dir, k.split()[0] + "_events.csv"), index=False)
    print("<<<COPY_FROM_HERE>>>")
    print("\n".join(L))
    print("<<<COPY_TO_HERE>>>")


def selftest():
    from step3b_baselines_m1 import features_all
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    n = 40
    x = rng.standard_normal((n, 18, 1000)).astype(np.float32)
    shared = rng.standard_normal((n, 1, 1000)).astype(np.float32)
    xc = x.copy()
    xc[:, 0] = shared[:, 0] + 0.05 * x[:, 0]
    xc[:, 8] = shared[:, 0] + 0.05 * x[:, 8]                                 # paire 0 : même signal des deux côtés
    f = coherence_features(xc).reshape(n, 2, 8, 5)
    check(f"paire synchrone : cohérence ~ 1 ({f[:, 0, 0].mean():.2f}) ; paires indépendantes : cohérence basse ({f[:, 0, 1:].mean():.2f})", f[:, 0, 0].mean() > 0.9 and f[:, 0, 1:].mean() < 0.3)
    xd = x.copy()
    xd[:, 8] = np.roll(shared[:, 0], 0, axis=-1) * 0 + x[:, 8]
    xd[:, 0] = shared[:, 0]
    xd[:, 8] = np.concatenate([np.zeros((n, 1), np.float32), shared[:, 0, :-1]], axis=1) * 0 + shared[:, 0]
    fi = coherence_features(xd).reshape(n, 2, 8, 5)
    check(f"signal identique sans décalage : cohérence imaginaire ~ 0 ({fi[:, 1, 0].mean():.2f}) malgré cohérence ~ 1 ({fi[:, 0, 0].mean():.2f}) (conduction de volume écartée)", fi[:, 1, 0].mean() < 0.1 and fi[:, 0, 0].mean() > 0.9)
    xm = xc[:, [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17]]
    check("cohérence des paires miroir identique si on échange les hémisphères", np.allclose(coherence_features(xm), coherence_features(xc), atol=1e-4))
    # pipeline complet sur un jeu où le bilatéral = hémisphères synchrones
    n_pat, nrec_pp, per_rec = 40, 3, 4
    n_rec = n_pat * nrec_pp
    N = n_rec * per_rec
    recs = np.repeat(np.arange(n_rec), per_rec)
    pat_of_rec = np.arange(n_rec) // nrec_pp
    cls = (np.arange(n_rec) % 3)[recs]
    X = rng.standard_normal((N, 18, 1000)).astype(np.float32) * 10
    sh = rng.standard_normal((N, 1000)).astype(np.float32) * 10
    for c, cols in ((0, range(0, 8)), (1, range(8, 16))):
        X[np.ix_(cls == c, list(cols))] *= 3
    for i in range(8):
        m = cls == 2
        X[m, i] = sh[m] + 0.2 * X[m, i]
        X[m, i + 8] = sh[m] + 0.2 * X[m, i + 8]
    role = np.where(pat_of_rec[recs] < 28, "train", "val")
    D = dict(x=X.astype(np.float16), m2=cls.astype(np.int64), y=np.ones(N, np.int64), pid=pat_of_rec[recs], rec=recs, eid=np.zeros(N, np.int64),
             valid=np.tile(np.ones(18, bool), (N, 1)), role=role, rec_names=np.array([f"rec{i:03d}" for i in range(n_rec)]),
             patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.tile(np.arange(per_rec) * 500, n_rec))
    _, _, bp, _ = features_all(D)
    Z = np.concatenate([np.asarray(bp, np.float32).reshape(N, -1), coherence_features(D["x"])], axis=1)
    check("matrice de variables : 90 (puissance) + 80 (cohérence) colonnes", Z.shape[1] == NPOW + 80)
    e_ref = pd.DataFrame(dict(recording=D["rec_names"], eid=0, pred_ref=(np.arange(n_rec) % 3)))
    L, evs = analyse(D, Z, e_ref, B=200)
    txt = "\n".join(L)
    check(f"4 lignes (V0, C1, C2, C3) et mêmes événements ({ {len(e) for e in evs.values()} } = {{{n_rec}}})", len(evs) == 4 and {len(e) for e in evs.values()} == {n_rec})
    check("rapport : 3 différences appariées", txt.count("différence appariée") == 3)
    c3 = [e for k, e in evs.items() if k.startswith("C3")][0]
    rb = float((c3[c3.y == 2].pred == 2).mean())
    check(f"jeu synchrone : la cohérence seule reconnaît le bilatéral (rappel {rb:.2f} > 0,8)", rb > 0.8)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--cv_dir", default=base + "lr_cv")
    ap.add_argument("--out_dir", default=base + "coherence_cv")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
