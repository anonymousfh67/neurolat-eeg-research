#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 19 -- Entraîner M2 directement sur la tâche « focal gauche / focal droit / généralisé ».  Validation croisée sur train + val SEULEMENT (ni eval, ni Siena).

Motivation (étape 18) : le « bilatéral » de la règle de comptage mélange des crises GÉNÉRALISÉES (type gnsz : fait clinique, retrouvé à 71 %, AUC 0,85) et des crises
FOCALES dont la règle donne « bilatéral » (741 événements, rappel 29 %, AUC 0,64 : artefact de la règle, non récupérable). Nouvelle tâche, décidée À L'AVANCE :
   classe 0 = focal gauche, classe 1 = focal droit (labels de la règle sur les événements fnsz/cpsz), classe 2 = généralisé (type gnsz seul, quel que soit le label de la règle).
   Les événements FOCAUX à règle bilatérale sont EXCLUS de l'entraînement et de l'évaluation (ils n'ont pas de côté défini) : c'est une autre tâche, pas un meilleur
   score de la même ; elle ne se compare qu'à elle-même.
Comparaison appariée sur les MÊMES événements :
   REF   = modèle B_asym80 de l'étape 7 (entraîné sur la règle à 3 classes), prédiction relue par l'application bilatéral -> généralisé (aucun ré-entraînement)
   N_B   = B_asym80 ré-entraîné sur la nouvelle tâche
   N_D   = D_hgb_B (arbres, paramètres fixés de l'étape 7) ré-entraîné sur la nouvelle tâche
Mêmes 5 plis par patient que l'étape 7, exactitude équilibrée par événement, IC95 par bootstrap de patients, différences appariées avec REF.
Deux comparaisons ; résultat exploratoire.

Usage :
  python step19_focal_generalized_cv.py --events_csv .../etape1_train_dev/events.csv
  python step19_focal_generalized_cv.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step7_features_m2_cv import patient_folds, fit_model, predict_model, N_FOLDS
from step4_train_m2 import m2_indices, event_aggregate
from step11_label_clarity import confusion_by_patient, boot_idx, boot_bacc, bacc_acc, ci, fmt

NAMES = ["focal G", "focal D", "généralisé"]
KEY = ["recording", "eid"]


def new_task_labels(D, events_csv):
    """Label de la nouvelle tâche par fenêtre : 0/1 focal gauche/droite, 2 généralisé, -1 exclu (focal à règle bilatérale ou inconnu)."""
    t = pd.read_csv(events_csv)
    hf = t["has_focal"].astype(str).str.lower().isin(["true", "1"]).values
    focal = {(r, int(e)): bool(h) for r, e, h in zip(t.recording, t.event_idx, hf)}
    names = D["rec_names"]
    y2 = np.full(len(D["m2"]), -1, np.int64)
    for i, (r, e, c) in enumerate(zip(D["rec"], D["eid"], D["m2"])):
        if c < 0 or e < 0 or (names[r], int(e)) not in focal:
            continue
        if not focal[(names[r], int(e))]:
            y2[i] = 2
        elif c in (0, 1):
            y2[i] = c
    return y2


def cv_new_task(name, bp, y2, pid, idx_all, n_folds=N_FOLDS):
    """Plis par patient définis sur TOUTES les fenêtres M2 d'origine (mêmes plis que l'étape 7) ; entraînement/test sur les fenêtres de la nouvelle tâche."""
    fold_of = patient_folds(pid[idx_all], n_folds)
    idx = idx_all[y2[idx_all] >= 0]
    fold = np.array([fold_of[p] for p in pid[idx]])
    P = np.zeros((len(idx), 3))
    for k in range(n_folds):
        tr, te = idx[fold != k], idx[fold == k]
        if len(te):
            P[fold == k] = predict_model(fit_model(name, bp, y2, pid, tr), bp[te])
    return idx, P


def ref_events(cv_dir, events_csv):
    p = os.path.join(cv_dir, "B_asym80_val_scores.csv")
    if not os.path.exists(p):
        raise SystemExit(f"ERREUR : {p} absent.")
    d = pd.read_csv(p)
    g = d.groupby(KEY).agg(patient=("patient", "first"), y=("y", "first"), pL=("pL", "mean"), pR=("pR", "mean"), pB=("pB", "mean")).reset_index()
    g["pred_ref"] = g[["pL", "pR", "pB"]].values.argmax(1)       # 0 G, 1 D, 2 bilatéral -> généralisé
    return g[KEY + ["pred_ref"]]


def analyse(D, bp, y2, ref, B=2000):
    idx_all = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx_all.sort()
    pid, rec, eid = D["pid"], D["rec"], D["eid"]
    evs = {}
    for tag, name in (("N_B B_asym80 réentraîné", "B_asym80"), ("N_D D_hgb_B réentraîné", "D_hgb_B")):
        idx, P = cv_new_task(name, bp, y2, pid, idx_all)
        e = event_aggregate(rec[idx], eid[idx], pid[idx], y2[idx], P)
        e["recording"] = D["rec_names"][e.rec.values]
        evs[tag] = e
    base = evs["N_B B_asym80 réentraîné"][KEY + ["pid", "y"]].merge(ref, on=KEY, how="left")
    n_lost = int(base.pred_ref.isna().sum())
    base = base[base.pred_ref.notna()]
    evs["REF B_asym80 (3 classes de la règle, relu)"] = base.rename(columns={"pred_ref": "pred"})
    keys = list(evs)
    keys = [keys[-1]] + keys[:-1]
    common = set(map(tuple, evs[keys[0]][KEY].values))
    for k in keys[1:]:
        common &= set(map(tuple, evs[k][KEY].values))
    for k in keys:
        e = evs[k]
        evs[k] = e[[tuple(x) in common for x in e[KEY].values]].sort_values(KEY).reset_index(drop=True)
    patients = sorted(set(evs[keys[0]].pid))
    bi = boot_idx(len(patients), B)
    C = {k: confusion_by_patient(evs[k].pid, evs[k].y, evs[k].pred, patients) for k in keys}
    cnt = [int((evs[keys[0]].y == c).sum()) for c in range(3)]
    lines = [f"{len(patients)} patients ; événements évalués {len(evs[keys[0]])} ({NAMES[0]} {cnt[0]} | {NAMES[1]} {cnt[1]} | {NAMES[2]} {cnt[2]}) ; événements de réf. sans correspondance : {n_lost}",
             "Exactitude équilibrée par événement, nouvelle tâche (les focaux à règle bilatérale sont exclus pour toutes les lignes) :"]
    for k in keys:
        lines.append(fmt(C[k], bi, k))
    b0 = boot_bacc(C[keys[0]], bi)
    for k in keys[1:]:
        d = boot_bacc(C[k], bi) - b0
        lo, hi = ci(d)
        dm = bacc_acc(C[k])[0] - bacc_acc(C[keys[0]])[0]
        v = "MEILLEUR (IC exclut 0)" if lo > 0 else ("moins bon (IC exclut 0)" if hi < 0 else "pas de différence détectable (IC contient 0)")
        lines.append(f"différence appariée {k} - REF : {dm:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}")
    lines.append("(rappels « G/D/B » des lignes ci-dessus = focal G / focal D / généralisé)")
    return lines, evs


def main(a):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    D = load_cache(a.cache_dir, a.split_csv, roles=("train", "val"), limit_per_role=a.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    y2 = new_task_labels(D, a.events_csv)
    lines, evs = analyse(D, bp, y2, ref_events(a.cv_dir, a.events_csv))
    os.makedirs(a.out_dir, exist_ok=True)
    for k, e in evs.items():
        e.to_csv(os.path.join(a.out_dir, k.split()[0] + "_events.csv"), index=False)
    print("<<<COPY_FROM_HERE>>>")
    print("\n".join(lines))
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test
def selftest():
    import tempfile, shutil
    from step3b_baselines_m1 import features_all
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    n_pat, nrec_pp, per_rec = 40, 4, 4
    n_rec = n_pat * nrec_pp
    n = n_rec * per_rec
    recs = np.repeat(np.arange(n_rec), per_rec)
    pat_of_rec = np.arange(n_rec) // nrec_pp
    kind = np.arange(n_rec) % 4           # 0 focal G, 1 focal D, 2 généralisé, 3 focal à règle bilatérale (bruit)
    kind_w = kind[recs]
    m2_rec = np.array([0, 1, 2, 2])[kind]         # label d'origine de la règle (3 classes)
    x = rng.standard_normal((n, 18, 1000)).astype(np.float32) * 10
    for kd, cols in ((0, range(0, 8)), (1, range(8, 16)), (2, range(0, 16))):
        x[np.ix_(kind_w == kd, list(cols))] *= 3
    role = np.where(pat_of_rec[recs] < 28, "train", "val")
    D = dict(x=x.astype(np.float16), m2=m2_rec[recs].astype(np.int64), y=np.ones(n, np.int64), pid=pat_of_rec[recs], rec=recs, eid=np.zeros(n, np.int64),
             valid=np.tile(np.ones(18, bool), (n, 1)), role=role, rec_names=np.array([f"rec{i:03d}" for i in range(n_rec)]),
             patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.tile(np.arange(per_rec) * 500, n_rec))
    tmp = tempfile.mkdtemp()
    try:
        evp = os.path.join(tmp, "events.csv")
        pd.DataFrame(dict(recording=D["rec_names"], event_idx=0, has_focal=(kind != 2))).to_csv(evp, index=False)
        y2 = new_task_labels(D, evp)
        wk = kind_w
        check("labels de la nouvelle tâche : G/D focaux -> 0/1, généralisé -> 2, focal à règle bilatérale -> exclu (-1)",
              (y2[wk == 0] == 0).all() and (y2[wk == 1] == 1).all() and (y2[wk == 2] == 2).all() and (y2[wk == 3] == -1).all())
        _, _, bp_all, _ = features_all(D)
        # scores de référence factices (modèle 3 classes relu) au format de l'étape 7
        e_ref = pd.DataFrame(dict(recording=D["rec_names"], eid=0, pred_ref=np.array([0, 1, 2, 2])[kind]))
        lines, evs = analyse(D, bp_all, y2, e_ref, B=200)
        txt = "\n".join(lines)
        sizes = {len(e) for e in evs.values()}
        check(f"mêmes événements dans toutes les lignes, sans les focaux à règle bilatérale ({sizes} = {{{n_rec * 3 // 4}}})", sizes == {n_rec * 3 // 4})
        check("rapport : 3 lignes d'exactitude et 2 différences appariées", txt.count("différence appariée") == 2 and txt.count("équilibrée") >= 3)
        nb = [e for k, e in evs.items() if k.startswith("N_B")][0]
        acc = float((nb.y == nb.pred).mean())
        check(f"jeu facile : la nouvelle tâche est bien apprise ({acc:.2f} > 0,9)", acc > 0.9)
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
    ap.add_argument("--out_dir", default=base + "focal_gen_cv")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
