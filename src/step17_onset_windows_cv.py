#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 17 -- Aligner le modèle sur la DÉFINITION du label : n'utiliser que le DÉBUT de la crise.  AUCUN accès à eval ni à Siena.

Constat. Le label (gauche / droite / bilatéral) est calculé sur les canaux actifs dans les 2 premières secondes de la crise (étape 1). Pourtant M2 est entraîné
sur TOUTES les fenêtres de la crise, et l'événement est décidé par la moyenne de TOUTES ses fenêtres. Une crise qui commence d'un seul côté s'étend souvent
aux deux hémisphères : ses fenêtres tardives ressemblent à du « bilatéral » mais portent le label « gauche » ou « droite » -> bruit d'étiquette possible.

Idée décidée À L'AVANCE (aucun réglage) : ne garder, pour chaque événement, que les fenêtres dont le début est à au plus T secondes de la première fenêtre
de l'événement, avec T dans {4, 10, 20} s (la fenêtre vaut 4 s, le pas 2 s). Deux familles, toutes rapportées :
  TA_T  : entraînement sur toutes les fenêtres, DÉCISION de l'événement sur les fenêtres des T premières secondes seulement (aucun ré-entraînement)
  TT_T  : entraînement ET décision sur les fenêtres des T premières secondes
Référence V0 = B_asym80 sur toutes les fenêtres (étape 7). Mêmes 5 plis par patient, exactitude équilibrée PAR ÉVÉNEMENT, IC95 par bootstrap de patients,
différences APPARIÉES avec V0. 6 comparaisons : un IC qui exclut 0 de peu reste un indice, pas une preuve. Résultat exploratoire.

Usage :
  python step17_onset_windows_cv.py [--limit_per_role N]
  python step17_onset_windows_cv.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step7_features_m2_cv import patient_folds, fit_model, predict_model, N_FOLDS
from step4_train_m2 import m2_indices, event_aggregate
from step11_label_clarity import confusion_by_patient, boot_idx, boot_bacc, bacc_acc, ci, fmt

FS = 250
T_GRID = (4, 10, 20)
MODEL = "B_asym80"


def rel_seconds(rec, eid, start, idx):
    """Pour chaque fenêtre de idx : secondes écoulées depuis la première fenêtre de son événement."""
    df = pd.DataFrame(dict(rec=rec[idx], eid=eid[idx], s=start[idx]))
    return ((df.s - df.groupby(["rec", "eid"]).s.transform("min")) / FS).values


def cv_onset(bp, y, pid, rec, eid, start, idx, n_folds=N_FOLDS):
    """Retourne {étiquette: tableau d'événements avec prédiction}."""
    rel = rel_seconds(rec, eid, start, idx)
    fold_of = patient_folds(pid[idx], n_folds)
    fold_all = np.array([fold_of[p] for p in pid[idx]])
    out = {}
    # modèle entraîné sur toutes les fenêtres (V0 et TA_T)
    P_all = np.zeros((len(idx), 3))
    for k in range(n_folds):
        tr, te = idx[fold_all != k], idx[fold_all == k]
        if len(te):
            P_all[fold_all == k] = predict_model(fit_model(MODEL, bp, y, pid, tr), bp[te])
    out["V0 toutes fenêtres"] = event_aggregate(rec[idx], eid[idx], pid[idx], y[idx], P_all)
    for T in T_GRID:
        m = rel <= T
        out[f"TA_{T} décision sur {T} s"] = event_aggregate(rec[idx][m], eid[idx][m], pid[idx][m], y[idx][m], P_all[m])
    # modèle entraîné sur les fenêtres de début seulement (TT_T)
    for T in T_GRID:
        m = rel <= T
        iT = idx[m]
        fT = fold_all[m]
        P = np.zeros((len(iT), 3))
        for k in range(n_folds):
            tr, te = iT[fT != k], iT[fT == k]
            if len(te):
                P[fT == k] = predict_model(fit_model(MODEL, bp, y, pid, tr), bp[te])
        out[f"TT_{T} entraînement + décision sur {T} s"] = event_aggregate(rec[iT], eid[iT], pid[iT], y[iT], P)
    return out, rel


def analyse(D, bp, B=2000):
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx.sort()
    y, pid, rec, eid, start = D["m2"], D["pid"], D["rec"], D["eid"], D["start"]
    evs, rel = cv_onset(bp, y, pid, rec, eid, start, idx)
    patients = sorted(set(pid[idx]))
    bi = boot_idx(len(patients), B)
    keys = sorted(evs, key=lambda k: (k[0] != "V", k[:2], k))
    keys = ["V0 toutes fenêtres"] + [k for k in keys if k != "V0 toutes fenêtres"]
    n_ev = {k: len(e) for k, e in evs.items()}
    lines = [f"{len(idx)} fenêtres M2, {len(patients)} patients (train + val), {N_FOLDS} plis par patient ; modèle {MODEL}",
             f"fenêtres conservées selon T (part des {len(idx)}) : " + " | ".join(f"T={T} s : {(rel <= T).mean():.0%}" for T in T_GRID),
             f"événements par variante : {sorted(set(n_ev.values()))} (doit être un seul nombre)", "Exactitude équilibrée par événement :"]
    C = {k: confusion_by_patient(evs[k].pid, evs[k].y, evs[k].pred, patients) for k in keys}
    for k in keys:
        lines.append(fmt(C[k], bi, k))
    b0 = boot_bacc(C[keys[0]], bi)
    res = {}
    for k in keys[1:]:
        d = boot_bacc(C[k], bi) - b0
        lo, hi = ci(d)
        dm = bacc_acc(C[k])[0] - bacc_acc(C[keys[0]])[0]
        v = "MEILLEUR (IC exclut 0)" if lo > 0 else ("moins bon (IC exclut 0)" if hi < 0 else "pas de différence détectable (IC contient 0)")
        lines.append(f"différence appariée {k} - V0 : {dm:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}")
        res[k] = (dm, lo, hi)
    return lines, evs, res


def main(a):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    D = load_cache(a.cache_dir, a.split_csv, roles=("train", "val"), limit_per_role=a.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    lines, evs, _ = analyse(D, bp)
    os.makedirs(a.out_dir, exist_ok=True)
    for k, e in evs.items():
        e.to_csv(os.path.join(a.out_dir, k.split()[0] + "_events.csv"), index=False)
    print("<<<COPY_FROM_HERE>>>")
    print("\n".join(lines))
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test
def selftest():
    from step3b_baselines_m1 import features_all
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    # position dans l'événement
    rec = np.repeat(np.arange(2), 4); eid = np.zeros(8, int); start = np.tile(np.arange(4) * 500, 2) + np.repeat([0, 10000], 4)
    r = rel_seconds(rec, eid, start, np.arange(8))
    check("secondes depuis la première fenêtre de l'événement : 0, 2, 4, 6 pour chaque événement", np.allclose(r, np.tile([0, 2, 4, 6], 2)))
    # jeu synthétique : les crises latérales COMMENCENT d'un côté puis deviennent bilatérales ; le label est celui du début
    n_pat, nrec_pp, per_rec = 30, 3, 8
    n_rec = n_pat * nrec_pp
    n = n_rec * per_rec
    recs = np.repeat(np.arange(n_rec), per_rec)
    pat_of_rec = np.arange(n_rec) // nrec_pp
    cls_rec = np.arange(n_rec) % 3
    cls = cls_rec[recs]
    k_in = np.tile(np.arange(per_rec), n_rec)          # rang de la fenêtre dans l'événement
    x = rng.standard_normal((n, 18, 1000)).astype(np.float32) * 10
    for c, cols in ((0, range(0, 8)), (1, range(8, 16))):
        early = (cls == c) & (k_in < 2)
        late = (cls == c) & (k_in >= 2)
        x[np.ix_(early, list(cols))] *= 3                # début : un seul côté
        x[np.ix_(late, list(range(0, 16)))] *= 3          # fin : les deux côtés (propagation)
    x[np.ix_(cls == 2, list(range(0, 16)))] *= 3
    role = np.where(pat_of_rec[recs] < 20, "train", "val")
    D = dict(x=x.astype(np.float16), m2=cls.astype(np.int64), y=np.ones(n, np.int64), pid=pat_of_rec[recs], rec=recs, eid=np.zeros(n, np.int64),
             valid=np.tile(np.ones(18, bool), (n, 1)), role=role, rec_names=np.array([f"rec{i:03d}" for i in range(n_rec)]),
             patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.tile(np.arange(per_rec) * 500, n_rec))
    _, _, bp_all, _ = features_all(D)
    lines, evs, res = analyse(D, bp_all, B=200)
    txt = "\n".join(lines)
    check(f"7 variantes (V0 + 3 TA + 3 TT) et un seul nombre d'événements ({len(evs)})", len(evs) == 7 and len({len(e) for e in evs.values()}) == 1 and len(evs["V0 toutes fenêtres"]) == n_rec)
    check("rapport complet : 6 différences appariées", txt.count("différence appariée") == 6)
    acc = {k: float((e.y == e.pred).mean()) for k, e in evs.items()}
    tt4 = [v for k, v in acc.items() if k.startswith("TT_4")][0]
    check(f"propagation simulée : entraîner/décider sur le début (TT_4 {tt4:.2f}) bat toutes les fenêtres (V0 {acc['V0 toutes fenêtres']:.2f})", tt4 > acc["V0 toutes fenêtres"] + 0.02)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--out_dir", default=base + "onset_cv")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
