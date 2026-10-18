#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 13 -- Combiner les modèles M2 de l'étape 7 (moyenne des probabilités) : gagne-t-on en gauche / droite / bilatéral ?  AUCUN accès à eval, AUCUN entraînement.

Les scores hors pli (validation croisée par patient, mêmes plis pour tous) de A_lr90, B_asym80, C_compact et D_hgb_B sont déjà écrits dans lr_cv/.
Deux combinaisons décidées À L'AVANCE : « tous » (moyenne des 4) et « B+D » (le modèle linéaire et les arbres, les plus différents). Référence = B_asym80 (le meilleur
modèle seul de l'étape 7 : comparer à lui est donc conservateur). Une combinaison n'est dite meilleure que si l'IC95 (bootstrap apparié par patients) de la
différence d'exactitude équilibrée PAR ÉVÉNEMENT exclut 0.

Usage :
  python step13_ensemble_m2_cv.py [--cv_dir .../lr_cv]
  python step13_ensemble_m2_cv.py --selftest
"""
import argparse, os, sys, tempfile, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step11_label_clarity import confusion_by_patient, boot_idx, boot_bacc, bacc_acc, ci, fmt

SINGLES = ["A_lr90", "B_asym80", "C_compact", "D_hgb_B"]
COMBOS = {"tous(A+B+C+D)": SINGLES, "B+D": ["B_asym80", "D_hgb_B"]}
KEY = ["recording", "eid"]


def load_scores(cv_dir):
    out = {}
    for n in SINGLES:
        p = os.path.join(cv_dir, f"{n}_val_scores.csv")
        if not os.path.exists(p):
            raise SystemExit(f"ERREUR : {p} absent : lancer step7_features_m2_cv.py d'abord.")
        out[n] = pd.read_csv(p).sort_values(KEY + ["start_sample"]).reset_index(drop=True)
    ref = out[SINGLES[0]]
    for n in SINGLES[1:]:
        if len(out[n]) != len(ref) or not (out[n][KEY + ["start_sample", "y"]].values == ref[KEY + ["start_sample", "y"]].values).all():
            raise SystemExit(f"ERREUR : {n} n'a pas les mêmes fenêtres que {SINGLES[0]}.")
    return out


def to_events(df, P):
    d = df[KEY + ["patient", "y"]].copy()
    d[["pL", "pR", "pB"]] = P
    g = d.groupby(KEY).agg(patient=("patient", "first"), y=("y", "first"), pL=("pL", "mean"), pR=("pR", "mean"), pB=("pB", "mean")).reset_index()
    g["pred"] = g[["pL", "pR", "pB"]].values.argmax(1)
    return g


def analyse(scores, B=2000):
    ref = scores[SINGLES[0]]
    probs = {n: scores[n][["pL", "pR", "pB"]].values for n in SINGLES}
    for c, members in COMBOS.items():
        probs[c] = np.mean([probs[m] for m in members], axis=0)
    evs = {n: to_events(ref, P) for n, P in probs.items()}
    patients = sorted(set(evs["B_asym80"].patient))
    bi = boot_idx(len(patients), B)
    C = {n: confusion_by_patient(e.patient, e.y, e.pred, patients) for n, e in evs.items()}
    lines = [f"{len(evs['B_asym80'])} événements, {len(patients)} patients (train + val), scores hors pli", "Exactitude par événement (labels d'origine) :"]
    for n in SINGLES + list(COMBOS):
        lines.append(fmt(C[n], bi, n))
    b0 = boot_bacc(C["B_asym80"], bi)
    res = {}
    for c in COMBOS:
        d = boot_bacc(C[c], bi) - b0
        lo, hi = ci(d)
        dm = bacc_acc(C[c])[0] - bacc_acc(C["B_asym80"])[0]
        v = f"{c} MEILLEUR (IC exclut 0)" if lo > 0 else (f"{c} moins bon (IC exclut 0)" if hi < 0 else "pas de différence détectable (IC contient 0)")
        lines.append(f"différence appariée {c} - B_asym80 (équilibrée) : {dm:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}")
        res[c] = (dm, lo, hi)
    return lines, evs, res


def main(a):
    lines, evs, _ = analyse(load_scores(a.cv_dir))
    print("<<<COPY_FROM_HERE>>>")
    print("\n".join(lines))
    print("<<<COPY_TO_HERE>>>")


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    n_pat, n_ev, w = 30, 4, 3
    rows = []
    for p in range(n_pat):
        for e in range(n_ev):
            y = int(rng.integers(0, 3))
            for k in range(w):
                rows.append((f"r{p:02d}", e, k * 500, f"p{p:02d}", y))
    base = pd.DataFrame(rows, columns=["recording", "eid", "start_sample", "patient", "y"])
    tmp = tempfile.mkdtemp()
    try:
        def noisy(sig):
            P = np.full((len(base), 3), 0.1)
            P[np.arange(len(base)), base.y.values] += 0.4
            P += rng.normal(0, sig, P.shape)
            P = np.clip(P, 1e-3, None)
            return P / P.sum(1, keepdims=True)
        for n, s in zip(SINGLES, (0.35, 0.35, 0.35, 0.35)):   # quatre modèles de même qualité, erreurs indépendantes : la moyenne doit aider
            d = base.copy()
            P = noisy(s)
            d[["pL", "pR", "pB"]] = P
            d.to_csv(os.path.join(tmp, f"{n}_val_scores.csv"), index=False)
        sc = load_scores(tmp)
        lines, evs, res = analyse(sc, B=300)
        acc = {n: float((e.y == e.pred).mean()) for n, e in evs.items()}
        check(f"{len(evs['B_asym80'])} événements = {n_pat * n_ev}", len(evs["B_asym80"]) == n_pat * n_ev)
        check(f"modèles indépendants de même qualité : la moyenne des 4 est meilleure ({acc['tous(A+B+C+D)']:.2f} > {acc['B_asym80']:.2f})", acc["tous(A+B+C+D)"] > acc["B_asym80"])
        check("rapport : 6 lignes de modèles et 2 différences appariées", sum("différence appariée" in l for l in lines) == 2 and sum("équilibrée" in l for l in lines) >= 6)
        # moyenne de probabilités = vraie moyenne
        P0 = sc["A_lr90"][["pL", "pR", "pB"]].values; P1 = sc["B_asym80"][["pL", "pR", "pB"]].values
        check("la combinaison B+D est la moyenne arithmétique des probabilités", np.allclose(np.mean([P1, sc["D_hgb_B"][["pL", "pR", "pB"]].values], axis=0).sum(1), 1))
        # fenêtres différentes -> refus
        d = pd.read_csv(os.path.join(tmp, "C_compact_val_scores.csv")).iloc[:-1]
        d.to_csv(os.path.join(tmp, "C_compact_val_scores.csv"), index=False)
        try:
            load_scores(tmp)
            check("fenêtres différentes entre modèles refusées", False)
        except SystemExit:
            check("fenêtres différentes entre modèles refusées", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_dir", default="/content/drive/MyDrive/projet_propre/lr_cv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
