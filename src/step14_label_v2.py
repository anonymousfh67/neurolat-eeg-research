#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 14 -- Seconde définition du label gauche / droite / bilatéral (règle PROPORTIONNELLE), à côté de l'ancienne (règle à seuils). AUCUN accès à eval.

Défaut logique de l'ancienne règle (étape 1) : « bilatéral dès que chaque côté a >= 2 canaux ». Une crise à 8 canaux à gauche et 2 à droite est donc étiquetée
bilatérale alors qu'elle est à dominante gauche. La règle v2 est proportionnelle, avec les MÊMES comptes de canaux (2 premières secondes, canaux distincts) :
    indice de latéralité  LI = (nG - nD) / (nG + nD)
    gauche si LI >= 0,5 (un côté a au moins 3 fois plus de canaux) ; droite si LI <= -0,5 ; sinon bilatéral ; aucun canal -> sans label.
Seuil 0,5 fixé À L'AVANCE, pas de réglage, pas de dépendance au type de crise (une seule règle, symétrique gauche/droite).
Les deux définitions sont conservées et rapportées. ATTENTION : ce sont deux TÂCHES différentes ; les scores ne sont pas comparables comme « meilleur / moins
bon » (le chiffre d'une tâche plus facile n'est pas une amélioration du modèle). Le script donne donc aussi : combien d'événements changent de classe, la part
de cas limites (stabilité aux seuils) de chaque définition, et le niveau de hasard.

Analyses (validation croisée PAR PATIENT, train + val, mêmes plis, modèles B_asym80 et A_lr90 de l'étape 7, comme avant) :
  1. table croisée ancienne x nouvelle définition ; 2. stabilité (v1 : 9 règles k x rho, comme l'étape 11 ; v2 : seuil LI dans {1/3, 0,5, 2/3}) ;
  3. performances par événement avec chaque définition (équilibrée, rappels G/D/B, IC95 par patients).

Usage :
  python step14_label_v2.py --events_csv <étape 1>/events.csv [--models B_asym80,A_lr90]
  python step14_label_v2.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_inventaire_labels import apply_rule, REF
from step7_features_m2_cv import cross_validate
from step4_train_m2 import m2_indices, event_aggregate
from step11_label_clarity import is_clear, confusion_by_patient, boot_idx, boot_bacc, bacc_acc, ci, fmt

LI_THRESHOLD = 0.5
LI_GRID = (1 / 3, 0.5, 2 / 3)
CODE = {"left": 0, "right": 1, "bilateral": 2}
NAMES = ["gauche", "droite", "bilatéral"]


def label_v2(nL, nR, thr=LI_THRESHOLD):
    n = int(nL) + int(nR)
    if n == 0:
        return None
    li = (int(nL) - int(nR)) / n
    if li >= thr - 1e-12:
        return "left"
    if li <= -thr + 1e-12:
        return "right"
    return "bilateral"


def is_clear_v2(nL, nR):
    labs = {label_v2(nL, nR, t) for t in LI_GRID}
    return None if None in labs else len(labs) == 1


def events_table(events_csv):
    ev = pd.read_csv(events_csv)
    need = {"recording", "event_idx", "has_focal", "nL_channels", "nR_channels"}
    if not need <= set(ev.columns):
        raise SystemExit(f"ERREUR : colonnes manquantes dans {events_csv} : {sorted(need - set(ev.columns))}")
    hf = ev["has_focal"].astype(str).str.lower().isin(["true", "1"])
    v1 = [apply_rule(h, int(l), int(r), REF["k"], REF["rho"], REF["tie"]) for h, l, r in zip(hf, ev.nL_channels, ev.nR_channels)]
    v2 = [label_v2(l, r) for l, r in zip(ev.nL_channels, ev.nR_channels)]
    c1 = [is_clear(h, l, r) for h, l, r in zip(hf, ev.nL_channels, ev.nR_channels)]
    c2 = [is_clear_v2(l, r) for l, r in zip(ev.nL_channels, ev.nR_channels)]
    return pd.DataFrame(dict(recording=ev.recording, event_idx=ev.event_idx.astype(int), v1=v1, v2=v2, clear1=c1, clear2=c2))


def window_labels(D, tab, col):
    m = {(r, e): CODE[v] for r, e, v in zip(tab.recording, tab.event_idx, tab[col]) if v is not None}
    names = D["rec_names"]
    return np.array([m.get((names[r], int(e)), -1) if e >= 0 else -1 for r, e in zip(D["rec"], D["eid"])], np.int64)


def analyse(D, bp, tab, models, B=2000):
    y1, y2 = window_labels(D, tab, "v1"), window_labels(D, tab, "v2")
    check_old = D["m2"]
    mism = int(((y1 != check_old) & (check_old >= 0)).sum())
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx.sort()
    keep = idx[(y1[idx] >= 0) & (y2[idx] >= 0)]
    patients = sorted(set(D["pid"][keep]))
    bi = boot_idx(len(patients), B)
    lines = [f"{len(keep)} fenêtres, {len(patients)} patients (train + val) ; fenêtres où l'ancienne règle recalculée diffère du cache : {mism} (doit être 0)"]
    ev = pd.DataFrame(dict(rec=D["rec"][keep], eid=D["eid"][keep], pid=D["pid"][keep], y1=y1[keep], y2=y2[keep])).drop_duplicates(["rec", "eid"])
    ct = pd.crosstab(ev.y1.map(dict(enumerate(NAMES))), ev.y2.map(dict(enumerate(NAMES))))
    ct = ct.reindex(index=NAMES, columns=NAMES, fill_value=0)
    lines += ["", f"1. Événements (ancienne définition en lignes, nouvelle en colonnes) ; {int((ev.y1 != ev.y2).sum())}/{len(ev)} changent de classe ({(ev.y1 != ev.y2).mean():.1%}) :", ct.to_string()]
    lines.append("   répartition des classes : ancienne " + " | ".join(f"{NAMES[c]} {(ev.y1 == c).mean():.0%}" for c in range(3)) + "  ;  nouvelle " + " | ".join(f"{NAMES[c]} {(ev.y2 == c).mean():.0%}" for c in range(3)))
    t = tab.set_index(["recording", "event_idx"])
    names = D["rec_names"]
    cl1 = np.array([t.clear1.get((names[r], int(e)), np.nan) for r, e in zip(ev.rec, ev.eid)], float)
    cl2 = np.array([t.clear2.get((names[r], int(e)), np.nan) for r, e in zip(ev.rec, ev.eid)], float)
    lines += ["", f"2. Stabilité du label aux seuils : événements dont le label ne change jamais : ancienne règle {np.nanmean(cl1):.1%} (9 règles k x rho) ; nouvelle règle {np.nanmean(cl2):.1%} (seuil LI 1/3, 1/2, 2/3)"]
    lines += ["", "3. Validation croisée par patient, événements, labels de chaque définition (hasard de l'exactitude équilibrée = 0,333) :"]
    res = {}
    for name in models:
        for tag, y in (("ancienne", y1), ("nouvelle", y2)):
            yy = np.where(np.isin(np.arange(len(y)), keep), y, -1)
            P, fold = cross_validate(bp, yy, D["pid"], keep, names=[name])
            e = event_aggregate(D["rec"][keep], D["eid"][keep], D["pid"][keep], yy[keep], P[name])
            C = confusion_by_patient(e.pid, e.y, e.pred, patients)
            lines.append(fmt(C, bi, f"{name} / définition {tag}"))
            res[(name, tag)] = e
    return lines, res


def main(a):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    tab = events_table(a.events_csv)
    D = load_cache(a.cache_dir, a.split_csv, roles=("train", "val"), limit_per_role=a.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    lines, res = analyse(D, bp, tab, [m for m in a.models.split(",") if m])
    os.makedirs(a.out_dir, exist_ok=True)
    tab.to_csv(os.path.join(a.out_dir, "labels_v1_v2.csv"), index=False)
    for (n, t), e in res.items():
        e.to_csv(os.path.join(a.out_dir, f"{n}_{t}_events.csv"), index=False)
    print("<<<COPY_FROM_HERE>>>")
    print("\n".join(lines))
    print("<<<COPY_TO_HERE>>>")


def selftest():
    import tempfile, shutil
    from step3b_baselines_m1 import features_all
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    check("(8,2) : gauche (l'ancienne règle disait bilatéral)", label_v2(8, 2) == "left" and apply_rule(False, 8, 2, REF["k"], REF["rho"], REF["tie"]) == "bilateral")
    check("(2,8) : droite", label_v2(2, 8) == "right")
    check("(4,4) et (2,2) : bilatéral ; (2,1) : bilatéral ; (3,1) : gauche (LI = 0,5) ; (1,0) : gauche", label_v2(4, 4) == label_v2(2, 2) == label_v2(2, 1) == "bilateral" and label_v2(3, 1) == "left" and label_v2(1, 0) == "left")
    check("(0,0) : sans label", label_v2(0, 0) is None)
    flip = {"left": "right", "right": "left", "bilateral": "bilateral"}
    sym = all(label_v2(a, b) is None if a + b == 0 else label_v2(b, a) == flip[label_v2(a, b)] for a in range(9) for b in range(9))
    check("symétrie gauche/droite exacte pour tous les comptes de 0 à 8 (miroir = échange des classes)", sym)
    check("(8,2) est stable aux seuils 1/3, 1/2, 2/3 ? non : LI = 0,6 < 2/3 -> limite ; (8,0) stable", is_clear_v2(8, 2) is False and is_clear_v2(8, 0) is True and is_clear_v2(0, 0) is None)
    # jeu synthétique : 30 patients x 3 enregistrements, un événement par enregistrement ; comptes tirés pour que v1 et v2 diffèrent parfois
    rng = np.random.default_rng(0)
    n_pat, nrec_pp, bg, ew = 30, 3, 8, 4
    n_rec = n_pat * nrec_pp
    per = bg + ew
    rec = np.repeat(np.arange(n_rec), per)
    pat_rec = np.arange(n_rec) // nrec_pp
    nL = rng.integers(0, 9, n_rec); nR = rng.integers(0, 9, n_rec)
    nL[(nL == 0) & (nR == 0)] = 1
    hf = np.zeros(n_rec, bool)
    tab = pd.DataFrame(dict(recording=[f"rec{i:03d}" for i in range(n_rec)], event_idx=0, has_focal=hf, nL_channels=nL, nR_channels=nR))
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "events.csv")
        tab.to_csv(p, index=False)
        T = events_table(p)
        v1 = np.array([CODE[v] for v in T.v1]); v2 = np.array([CODE[v] for v in T.v2])
        x = rng.standard_normal((n_rec * per, 18, 1000)).astype(np.float32) * 10
        y = np.zeros(n_rec * per, int); m2 = np.full(n_rec * per, -1); eid = np.full(n_rec * per, -1)
        for r in range(n_rec):
            sz = slice(r * per + bg, (r + 1) * per)
            x[sz][:, :8] *= (1 + nL[r] / 3.0); x[sz][:, 8:16] *= (1 + nR[r] / 3.0)    # l'amplitude suit le nombre de canaux atteints
            y[sz] = 1; m2[sz] = v1[r]; eid[sz] = 0
        role = np.where(pat_rec[rec] < 20, "train", "val")
        D = dict(x=x.astype(np.float16), m2=m2, y=y, pid=pat_rec[rec], rec=rec, eid=eid, valid=np.tile(np.ones(18, bool), (len(rec), 1)), role=role,
                 rec_names=np.array(tab.recording), patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.arange(len(rec)))
        wl = window_labels(D, T, "v2")
        check("labels de fenêtres v2 = label de l'événement ; le fond (eid = -1) reste -1", np.array_equal(wl[eid >= 0], np.repeat(v2, ew)) and (wl[eid < 0] == -1).all())
        _, _, bp, _ = features_all(D)
        lines, res = analyse(D, bp, T, ["B_asym80"], B=200)
        txt = "\n".join(lines)
        check("l'ancienne règle recalculée est identique à celle du cache (0 écart)", "diffère du cache : 0 " in txt)
        check("rapport complet : table croisée, stabilité, deux définitions évaluées", all(s in txt for s in ("changent de classe", "Stabilité", "définition ancienne", "définition nouvelle")))
        check("un événement par enregistrement dans chaque évaluation", all(len(e) == n_rec for e in res.values()))
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
    ap.add_argument("--out_dir", default=base + "label_v2")
    ap.add_argument("--models", default="B_asym80,A_lr90")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
