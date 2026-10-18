#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 18 -- OÙ se trompe le « bilatéral » ?  Diagnostic SANS entraînement, SANS eval, SANS Siena.

Dans TUSZ, un événement est soit FOCAL (annotation fnsz / cpsz), soit GÉNÉRALISÉ SEUL (gnsz, type de crise cliniquement bilatéral dès le début : c'est le
concept ILAE « début généralisé »). Notre label « bilatéral » mélange deux choses très différentes :
   (a) des crises GÉNÉRALISÉES (gnsz) dont la règle de canaux dit « bilatéral » ;
   (b) des crises FOCALES dont la règle de canaux (>= 2 canaux de chaque côté, ou égalité) dit « bilatéral ».
Question : le modèle (B_asym80, scores hors pli de l'étape 7) échoue-t-il surtout sur (b), c'est-à-dire sur un label qui est une conséquence de la règle
de comptage de canaux et non un fait clinique ? Si oui, une tâche « focal gauche / focal droit / généralisé » (type de crise + règle de côté) serait plus
apprenable ET plus proche de la classification ILAE ; ce serait une AUTRE tâche (pas un meilleur score sur la même), à annoncer comme telle.

Sorties : tableau label x type (événements, patients), rappel du modèle par cellule, AUC de la probabilité « bilatéral » pour (généralisé vs focal) et
(bilatéral focal vs unilatéral focal) avec IC95 par bootstrap de patients, et rappels de la tâche alternative lue sur les mêmes prédictions (diagnostic seulement,
le modèle n'a pas été entraîné pour elle).

Usage :
  python step18_bilateral_by_type.py --events_csv .../etape1_train_dev/events.csv [--cv_dir .../lr_cv]
  python step18_bilateral_by_type.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_inventaire_labels import apply_rule

KEY = ["recording", "eid"]
NAMES = ["gauche", "droite", "bilatéral"]
REF = dict(k=2, rho=2.0, tie="bilateral")


def load_events(cv_dir, model="B_asym80"):
    p = os.path.join(cv_dir, f"{model}_val_scores.csv")
    if not os.path.exists(p):
        raise SystemExit(f"ERREUR : {p} absent.")
    d = pd.read_csv(p)
    g = d.groupby(KEY).agg(patient=("patient", "first"), y=("y", "first"), pL=("pL", "mean"), pR=("pR", "mean"), pB=("pB", "mean")).reset_index()
    g["pred"] = g[["pL", "pR", "pB"]].values.argmax(1)
    return g


def add_type(ev, events_csv):
    t = pd.read_csv(events_csv)
    hf = t["has_focal"].astype(str).str.lower().isin(["true", "1"])
    t = pd.DataFrame(dict(recording=t.recording, eid=t.event_idx.astype(int), focal=hf.values))
    out = ev.merge(t, on=KEY, how="left")
    n_missing = int(out.focal.isna().sum())
    out = out[out.focal.notna()].copy()
    out["focal"] = out.focal.astype(bool)
    return out, n_missing


def auc(score, pos):
    """AUC de Mann-Whitney (pos : booléens) ; NaN si une classe est vide."""
    s = np.asarray(score, float)
    p = np.asarray(pos, bool)
    n1, n0 = p.sum(), (~p).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(df, score_col, pos, B=1000, seed=0):
    pats = sorted(set(df.patient))
    groups = {p: df.index[df.patient == p].values for p in pats}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        chosen = rng.choice(len(pats), len(pats))
        idx = np.concatenate([groups[pats[i]] for i in chosen])
        v = auc(df.loc[idx, score_col].values, pos.loc[idx].values)
        if not np.isnan(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def analyse(ev, n_missing=0):
    ev = ev.reset_index(drop=True)
    L = [f"{len(ev)} événements, {ev.patient.nunique()} patients (train + val, scores hors pli du modèle B_asym80) ; événements absents de events.csv : {n_missing}"]
    L.append("Événements par label de la règle x type de crise (événements / patients) :")
    L.append(f"  {'label':<10} {'focal':>16} {'généralisé seul':>18}")
    for c in range(3):
        row = []
        for foc in (True, False):
            s = ev[(ev.y == c) & (ev.focal == foc)]
            row.append(f"{len(s):>6} / {s.patient.nunique():<5}")
        L.append(f"  {NAMES[c]:<10} {row[0]:>16} {row[1]:>18}")
    L.append("Rappel du modèle (prédiction == label de la règle) par cellule :")
    for c in range(3):
        parts = []
        for foc, nm in ((True, "focal"), (False, "généralisé")):
            s = ev[(ev.y == c) & (ev.focal == foc)]
            parts.append(f"{nm} {np.mean(s.pred == c):.2f} (n={len(s)})" if len(s) else f"{nm} -")
        L.append(f"  {NAMES[c]:<10} " + " | ".join(parts))
    b = ev[ev.y == 2]
    L.append(f"Part des événements « bilatéral » qui sont GÉNÉRALISÉS seuls : {np.mean(~b.focal):.0%} ; qui sont FOCAUX à règle bilatérale : {np.mean(b.focal):.0%}")
    # AUC 1 : généralisé vs focal
    ev["gen"] = ~ev.focal
    a1 = auc(ev.pB, ev.gen)
    lo1, hi1 = boot_auc(ev, "pB", ev.gen)
    L.append(f"AUC de la probabilité « bilatéral » pour séparer GÉNÉRALISÉ seul de FOCAL (tous labels) : {a1:.3f} [{lo1:.3f}, {hi1:.3f}]")
    # AUC 2 : bilatéral focal vs unilatéral focal
    f = ev[ev.focal].reset_index(drop=True)
    a2 = auc(f.pB, f.y == 2)
    lo2, hi2 = boot_auc(f, "pB", f.y == 2)
    L.append(f"AUC de la probabilité « bilatéral » pour séparer, parmi les FOCAUX, bilatéral (règle) de gauche / droite : {a2:.3f} [{lo2:.3f}, {hi2:.3f}]")
    # tâche alternative lue sur les mêmes prédictions (diagnostic) : focal gauche / focal droit / généralisé (focaux à règle bilatérale exclus)
    alt = ev[~((ev.focal) & (ev.y == 2))].copy()
    alt["t"] = np.where(~alt.focal, 2, alt.y)
    rec = [np.mean(alt[alt.t == c].pred == c) if (alt.t == c).any() else np.nan for c in range(3)]
    n_alt = [int((alt.t == c).sum()) for c in range(3)]
    L.append("Tâche alternative « focal gauche / focal droit / généralisé » (focaux à règle bilatérale exclus), lue sur les MÊMES prédictions (diagnostic, pas entraîné pour elle) :")
    L.append(f"  rappels {rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f} | n = {n_alt[0]}/{n_alt[1]}/{n_alt[2]} | exactitude équilibrée {np.nanmean(rec):.3f}")
    return L, dict(auc_gen=a1, auc_foc=a2, rec_alt=rec)


def main(a):
    ev, miss = add_type(load_events(a.cv_dir), a.events_csv)
    L, _ = analyse(ev, miss)
    print("<<<COPY_FROM_HERE>>>")
    print("\n".join(L))
    print("<<<COPY_TO_HERE>>>")


def selftest():
    import tempfile, shutil
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    check("AUC parfaite = 1", auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0)
    check("AUC inversée = 0", auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0)
    check("AUC sans une classe = NaN", np.isnan(auc([0.1, 0.2], [1, 1])))
    rng = np.random.default_rng(0)
    n = 600
    pat = rng.integers(0, 40, n)
    kind = rng.choice(["L", "R", "Bfoc", "Bgen"], n, p=[0.3, 0.3, 0.2, 0.2])
    y = np.select([kind == "L", kind == "R"], [0, 1], 2)
    focal = kind != "Bgen"
    pB = np.where(kind == "Bgen", rng.beta(8, 2, n), np.where(kind == "Bfoc", rng.beta(2, 5, n), rng.beta(2, 8, n)))
    pL = (1 - pB) * np.where(kind == "L", 0.8, 0.2)
    pR = 1 - pB - pL
    ev = pd.DataFrame(dict(recording=[f"r{i}" for i in range(n)], eid=0, patient=[f"p{p}" for p in pat], y=y, pL=pL, pR=pR, pB=pB, focal=focal))
    ev["pred"] = ev[["pL", "pR", "pB"]].values.argmax(1)
    L, r = analyse(ev)
    txt = "\n".join(L)
    check(f"généralisé bien séparé du focal (AUC {r['auc_gen']:.2f} > 0,85) et bilatéral focal mal séparé (AUC {r['auc_foc']:.2f} < 0,75)", r["auc_gen"] > 0.85 and r["auc_foc"] < 0.75)
    check("rapport : tableau label x type, rappels par cellule, deux AUC, tâche alternative", all(s in txt for s in ("généralisé seul", "Rappel du modèle", "AUC", "Tâche alternative")))
    # jointure avec events.csv : types lus correctement
    tmp = tempfile.mkdtemp()
    try:
        pd.DataFrame(dict(recording=["a", "b"], event_idx=[0, 0], has_focal=[True, False])).to_csv(os.path.join(tmp, "e.csv"), index=False)
        e = pd.DataFrame(dict(recording=["a", "b", "c"], eid=[0, 0, 0], patient=["p", "q", "r"], y=[0, 2, 1], pL=.5, pR=.2, pB=.3, pred=[0, 2, 1]))
        out, miss = add_type(e, os.path.join(tmp, "e.csv"))
        check("jointure : types lus (a focal, b généralisé), événement absent compté", out.focal.tolist() == [True, False] and miss == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cv_dir", default=base + "lr_cv")
    ap.add_argument("--events_csv", default=base + "etape1_train_dev/events.csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
