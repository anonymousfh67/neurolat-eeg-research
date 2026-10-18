#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 16 -- Dernier essai EXPLORATOIRE pour le « bilatéral » (rappel ~0,52 contre ~0,9 pour gauche / droite). AUCUN accès à eval ni à Siena.

Deux idées, décidées À L'AVANCE, croisées en 4 variantes (toutes rapportées, qu'elles réussissent ou non) :
  V0 : B_asym80 (référence de l'étape 7), prédiction = classe de plus forte probabilité moyenne de l'événement.
  V1 : B_asym80 + POIDS SUR « BILATÉRAL » : on multiplie la probabilité bilatérale de l'événement par w avant le choix de la classe.
       w est choisi dans une grille fixe (0,8 ... 4) par validation croisée INTERNE (4 plis par patient, uniquement dans les patients
       d'entraînement du pli externe) en maximisant l'exactitude équilibrée par événement. Le pli externe n'intervient jamais dans le choix de w.
  V2 : B_absdiff = B_asym80 + |G-D| par paire miroir et par bande (40 variables de plus, invariantes au miroir : un modèle linéaire ne peut pas
       former « les deux côtés sont actifs » à partir de la seule différence signée).
  V3 : V2 + poids sur « bilatéral » (même procédure que V1).

Protocole : patients train + val, 5 plis externes par patient (mêmes plis que l'étape 7), exactitude équilibrée PAR ÉVÉNEMENT, IC95 par bootstrap
de patients, différences APPARIÉES avec V0. Une variante n'est dite meilleure que si l'IC de la différence exclut 0. Trois comparaisons : un IC qui
exclut 0 de peu reste un indice, pas une preuve (comparaisons multiples). Résultat à présenter comme EXPLORATOIRE : il n'existe plus de jeu de test vierge.

Usage :
  python step16_bilateral_cv.py [--limit_per_role N]
  python step16_bilateral_cv.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step7_features_m2_cv as s7
from step7_features_m2_cv import patient_folds, fit_model, predict_model, feat_B, _bp3, N_FOLDS
from step4_train_m2 import m2_indices, event_aggregate, m2_metrics
from step11_label_clarity import confusion_by_patient, boot_idx, boot_bacc, bacc_acc, ci, fmt

W_GRID = (0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
INNER_FOLDS = 4
NAMES = ["G", "D", "B"]


def feat_Babs(bp):
    b = _bp3(bp)
    d = np.abs(b[:, :8] - b[:, 8:16]).reshape(len(b), -1)
    return np.concatenate([feat_B(bp), d], axis=1)


s7.FEATURES["B_absdiff"] = feat_Babs


# ------------------------------------------------------------------ poids sur « bilatéral »
def bacc_with_weight(ev, w):
    P = ev[["pL", "pR", "pB"]].values * np.array([1.0, 1.0, w])
    return m2_metrics(ev.y.values, P.argmax(1), ev.pid.values)["bacc"]


def choose_w(ev, grid=W_GRID):
    """Poids maximisant l'exactitude équilibrée par événement ; en cas d'égalité, le plus proche de 1 (pas de correction sans gain)."""
    scores = [(bacc_with_weight(ev, w), -abs(np.log(w)), w) for w in grid]
    return max(scores)[2]


def window_folds(pid, idx, n_folds, seed=0):
    fold_of = patient_folds(pid[idx], n_folds, seed)
    return np.array([fold_of[p] for p in pid[idx]])


def cv_with_weight(name, bp, y, pid, rec, eid, idx, n_folds=N_FOLDS):
    """Retourne (probabilités hors pli par fenêtre, pli de chaque fenêtre, w choisi par pli externe)."""
    fold = window_folds(pid, idx, n_folds)
    P = np.zeros((len(idx), 3))
    w_of = {}
    for k in range(n_folds):
        tr, te = idx[fold != k], idx[fold == k]
        if len(te) == 0:
            continue
        # validation croisée interne, uniquement dans tr
        ifold = window_folds(pid, tr, INNER_FOLDS, seed=1)
        Pin = np.zeros((len(tr), 3))
        for j in range(INNER_FOLDS):
            a, b = tr[ifold != j], tr[ifold == j]
            if len(b):
                Pin[ifold == j] = predict_model(fit_model(name, bp, y, pid, a), bp[b])
        w_of[k] = choose_w(event_aggregate(rec[tr], eid[tr], pid[tr], y[tr], Pin))
        P[fold == k] = predict_model(fit_model(name, bp, y, pid, tr), bp[te])
    return P, fold, w_of


def events_with_pred(rec, eid, pid, y, P, fold, w_of=None):
    ev = event_aggregate(rec, eid, pid, y, P)
    if w_of is None:
        return ev
    fold_of_pid = dict(zip(pid, fold))
    w = np.array([w_of[fold_of_pid[p]] for p in ev.pid])
    ev["pred"] = (ev[["pL", "pR", "pB"]].values * np.stack([np.ones(len(ev)), np.ones(len(ev)), w], axis=1)).argmax(1)
    return ev


# ------------------------------------------------------------------ analyse
def analyse(D, bp, B=2000):
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx.sort()
    y, pid, rec, eid = D["m2"], D["pid"], D["rec"], D["eid"]
    patients = sorted(set(pid[idx]))
    bi = boot_idx(len(patients), B)
    lines = [f"{len(idx)} fenêtres M2, {len(patients)} patients (train + val), {N_FOLDS} plis externes par patient, {INNER_FOLDS} plis internes pour choisir w ; grille w = {W_GRID}"]
    evs, ws = {}, {}
    for name, tag in (("B_asym80", "V0 B_asym80"), ("B_absdiff", "V2 B_absdiff")):
        P, fold, w_of = cv_with_weight(name, bp, y, pid, rec, eid, idx)
        base = tag.split()[0]
        evs[tag] = events_with_pred(rec[idx], eid[idx], pid[idx], y[idx], P, fold)
        t2 = {"V0": "V1 B_asym80 + poids B", "V2": "V3 B_absdiff + poids B"}[base]
        evs[t2] = events_with_pred(rec[idx], eid[idx], pid[idx], y[idx], P, fold, w_of)
        ws[t2] = w_of
    order = sorted(evs)
    C = {t: confusion_by_patient(e.pid, e.y, e.pred, patients) for t, e in evs.items()}
    lines.append(f"Exactitude équilibrée par événement ({len(evs['V0 B_asym80'])} événements) :")
    for t in order:
        lines.append(fmt(C[t], bi, t))
    lines.append("poids w choisis par pli externe : " + " | ".join(f"{t} : " + ",".join(f"{ws[t][k]:g}" for k in sorted(ws[t])) for t in ws))
    b0 = boot_bacc(C["V0 B_asym80"], bi)
    res = {}
    for t in order[1:]:
        d = boot_bacc(C[t], bi) - b0
        lo, hi = ci(d)
        dm = bacc_acc(C[t])[0] - bacc_acc(C["V0 B_asym80"])[0]
        v = "MEILLEUR (IC exclut 0)" if lo > 0 else ("moins bon (IC exclut 0)" if hi < 0 else "pas de différence détectable (IC contient 0)")
        lines.append(f"différence appariée {t} - V0 : {dm:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}")
        res[t] = (dm, lo, hi)
    # rappel du bilatéral et coût sur gauche / droite
    lines.append("rappels G/D/B par variante (on regarde ce que gagne le bilatéral et ce que coûtent gauche / droite) :")
    for t in order:
        r = bacc_acc(C[t])[2]
        lines.append(f"  {t:<26} {r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f}")
    return lines, evs, res


def main(a):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    D = load_cache(a.cache_dir, a.split_csv, roles=("train", "val"), limit_per_role=a.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    lines, evs, _ = analyse(D, bp)
    os.makedirs(a.out_dir, exist_ok=True)
    for t, e in evs.items():
        e.to_csv(os.path.join(a.out_dir, t.split()[0] + "_events.csv"), index=False)
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
    # 1) variables |G-D| invariantes au miroir ; variables de B changent de signe
    bp = rng.standard_normal((20, 90)).astype(np.float32)
    mb = s7.mirror_bp(bp)
    check("|G-D| (40 variables ajoutées) identiques sur la fenêtre miroir", np.allclose(feat_Babs(bp)[:, 80:], feat_Babs(mb)[:, 80:]))
    check("feat_Babs a 120 variables = 80 (B) + 40 (|G-D|)", feat_Babs(bp).shape[1] == 120)
    # 2) choix de w : un bilatéral sous-estimé -> w > 1 ; des probabilités déjà calibrées -> w = 1 (égalité => plus proche de 1)
    n = 600
    y = np.tile([0, 1, 2], n // 3)
    pid = np.repeat(np.arange(20), n // 20)
    P = np.full((n, 3), 0.2)
    P[np.arange(n), y] += 0.3
    P[y == 2] = [0.35, 0.35, 0.30]                       # bilatéral sous-estimé : l'argmax se trompe, un poids modeste le corrige
    ev = pd.DataFrame(dict(y=y, pid=pid, pL=P[:, 0], pR=P[:, 1], pB=P[:, 2]))
    check(f"bilatéral sous-estimé : w choisi > 1 ({choose_w(ev)})", choose_w(ev) > 1)
    Q = np.full((n, 3), 0.2); Q[np.arange(n), y] += 0.3
    ev2 = pd.DataFrame(dict(y=y, pid=pid, pL=Q[:, 0], pR=Q[:, 1], pB=Q[:, 2]))
    check(f"probabilités déjà bonnes : w = 1 ({choose_w(ev2)})", choose_w(ev2) == 1.0)
    # 3) jeu synthétique complet
    n_pat, per_rec, nrec_pp = 30, 4, 3
    n_rec = n_pat * nrec_pp
    n_w = n_rec * per_rec
    recs = np.repeat(np.arange(n_rec), per_rec)
    pat_of_rec = np.arange(n_rec) // nrec_pp
    cls_rec = np.arange(n_rec) % 3
    cls = cls_rec[recs]
    x = rng.standard_normal((n_w, 18, 1000)).astype(np.float32) * 10
    for c, cols in ((0, range(0, 8)), (1, range(8, 16)), (2, range(0, 16))):
        x[np.ix_(cls == c, list(cols))] *= 3
    role = np.where(pat_of_rec[recs] < 20, "train", "val")
    D = dict(x=x.astype(np.float16), m2=cls.astype(np.int64), y=np.ones(n_w, np.int64), pid=pat_of_rec[recs], rec=recs, eid=np.zeros(n_w, np.int64),
             valid=np.tile(np.ones(18, bool), (n_w, 1)), role=role, rec_names=np.array([f"rec{i:03d}" for i in range(n_rec)]),
             patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.arange(n_w))
    _, _, bp_all, _ = features_all(D)
    lines, evs, res = analyse(D, bp_all, B=200)
    txt = "\n".join(lines)
    check(f"4 variantes rapportées ({len(evs)})", len(evs) == 4)
    check("un événement par enregistrement et mêmes événements dans toutes les variantes", all(len(e) == n_rec for e in evs.values())
          and all((e[["rec", "eid"]].values == evs["V0 B_asym80"][["rec", "eid"]].values).all() for e in evs.values()))
    check("rapport complet : exactitude, poids choisis, 3 différences appariées, rappels", all(s in txt for s in ("poids w choisis", "rappels G/D/B")) and txt.count("différence appariée") == 3)
    ws = [float(v) for l in lines if l.startswith("poids w choisis") for v in l.replace("|", ":").replace(",", ":").split(":")[1:] if v.strip().replace(".", "").isdigit()]
    check("tous les w choisis sont dans la grille", len(ws) > 0 and all(w in W_GRID for w in ws))
    acc = (evs["V0 B_asym80"].y == evs["V0 B_asym80"].pred).mean()
    check(f"jeu facile : V0 bien prédit ({acc:.2f} > 0,9)", acc > 0.9)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--out_dir", default=base + "bilateral_cv")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
