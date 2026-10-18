#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 11 -- Le plafond de la latéralisation (surtout « bilatéral ») vient-il du MODÈLE ou du LABEL ?  AUCUN accès à eval.

Idée. Le label d'une crise vient d'une règle à seuils (k canaux par côté, rapport rho). Une crise dont le label CHANGE quand on bouge un peu ces seuils est
une crise « limite » : même un expert pourrait la ranger autrement, un modèle ne peut pas la deviner. Une crise dont le label ne change jamais est « claire ».
Définition fixée À L'AVANCE (aucun réglage) : on recalcule le label avec k dans {1,2,3} et rho dans {1.5,2,3} (9 règles) à partir des comptes de canaux
du début de crise (events.csv de l'étape 1). Claire = les 9 règles donnent le même label ; limite = au moins un changement.

Trois analyses, toutes en validation croisée PAR PATIENT (5 plis) sur train + val, avec les labels d'origine pour l'évaluation :
  1. Part de cas limites par classe, et exactitude équilibrée par événement sur les cas clairs / limites / tous (E0 = entraînement sur tous les cas).
  2. E1 : entraîner sur les cas CLAIRS seulement (évaluation toujours sur tous les cas, mêmes labels), différence appariée E1 - E0 avec IC par patients.
  3. « Incertain » : on garde seulement les événements où le modèle est le plus confiant (couverture 100, 90, ... 50 %), exactitude sur les cas décidés.
Verdict automatique : E1 n'est dit meilleur que si l'IC de la différence appariée exclut 0.

Usage :
  python step11_label_clarity.py --events_csv <dossier étape 1>/events.csv [--models B_asym80,A_lr90]
  python step11_label_clarity.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_inventaire_labels import apply_rule
from step7_features_m2_cv import patient_folds, fit_model, predict_model, N_FOLDS
from step4_train_m2 import m2_indices, event_aggregate

K_GRID = (1, 2, 3)
RHO_GRID = (1.5, 2.0, 3.0)
CLASS = {"left": 0, "right": 1, "bilateral": 2}
NAMES = ["G", "D", "B"]
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
N_BOOT = 2000


# ------------------------------------------------------------------ clarté du label
def is_clear(has_focal, nL, nR):
    """True si le label est identique pour les 9 règles (k, rho) ; False sinon (cas limite) ; None si l'événement n'a pas de label."""
    labs = {apply_rule(bool(has_focal), int(nL), int(nR), k, rho, "bilateral") for k in K_GRID for rho in RHO_GRID}
    if None in labs:
        return None
    return len(labs) == 1


def clarity_table(events_csv):
    ev = pd.read_csv(events_csv)
    need = {"recording", "event_idx", "has_focal", "nL_channels", "nR_channels"}
    if not need <= set(ev.columns):
        raise SystemExit(f"ERREUR : colonnes manquantes dans {events_csv} : {sorted(need - set(ev.columns))}")
    hf = ev["has_focal"].astype(str).str.lower().isin(["true", "1"])
    cl = [is_clear(h, l, r) for h, l, r in zip(hf, ev.nL_channels, ev.nR_channels)]
    ev = ev.assign(clear=[(-1 if c is None else int(c)) for c in cl])
    return ev[["recording", "event_idx", "clear"]]


def window_clarity(D, table):
    """Pour chaque fenêtre de D : 1 = événement clair, 0 = limite, -1 = événement introuvable dans la table."""
    m = {(r, int(e)): int(c) for r, e, c in zip(table.recording, table.event_idx, table.clear)}
    names = D["rec_names"]
    return np.array([m.get((names[r], int(e)), -1) for r, e in zip(D["rec"], D["eid"])], np.int8)


# ------------------------------------------------------------------ métriques par patient (bootstrap)
def confusion_by_patient(pid, y, pred, patients):
    C = np.zeros((len(patients), 3, 3))
    pos = {p: i for i, p in enumerate(patients)}
    for p, t, q in zip(pid, y, pred):
        C[pos[p], int(t), int(q)] += 1
    return C


def bacc_acc(C):
    S = C.sum(0)
    tot = S.sum(1)
    rec = np.where(tot > 0, np.diag(S) / np.maximum(tot, 1), np.nan)
    acc = np.trace(S) / S.sum() if S.sum() > 0 else np.nan
    return (np.nanmean(rec) if (tot > 0).any() else np.nan), acc, rec


def boot_idx(n_pat, B=N_BOOT, seed=0):
    return np.random.default_rng(seed).integers(0, n_pat, (B, n_pat))


def boot_bacc(C, idx):
    out = np.empty(len(idx))
    for b, ii in enumerate(idx):
        out[b] = bacc_acc(C[ii])[0]
    return out


def ci(v):
    v = v[~np.isnan(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def fmt(C, idx, tag):
    b, a, r = bacc_acc(C)
    lo, hi = ci(boot_bacc(C, idx))
    n = int(C.sum())
    return f"  {tag:<34} équilibrée {b:.3f} [{lo:.3f}, {hi:.3f}] | exactitude {a:.3f} | rappels G/D/B {r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f} | n={n}"


# ------------------------------------------------------------------ validation croisée : tous les cas vs cas clairs
def cross_validate_pair(name, bp, y, pid, idx, clear, n_folds=N_FOLDS):
    fold_of = patient_folds(pid[idx], n_folds)
    fold = np.array([fold_of[p] for p in pid[idx]])
    P0 = np.zeros((len(idx), 3))
    P1 = np.zeros((len(idx), 3))
    for k in range(n_folds):
        tr, te = idx[fold != k], idx[fold == k]
        if len(te) == 0:
            continue
        P0[fold == k] = predict_model(fit_model(name, bp, y, pid, tr), bp[te])
        trc = tr[clear[tr] == 1]
        P1[fold == k] = predict_model(fit_model(name, bp, y, pid, trc), bp[te])
    return P0, P1


def analyse(D, bp, table, models, B=N_BOOT):
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx.sort()
    clear = window_clarity(D, table)
    n_unknown = int((clear[idx] < 0).sum())
    lines = [f"{len(idx)} fenêtres M2, {len(set(D['pid'][idx]))} patients (train + val), {N_FOLDS} plis par patient ; fenêtres sans événement dans events.csv : {n_unknown}"]
    patients = sorted(set(D["pid"][idx]))
    bi = boot_idx(len(patients), B)
    results = {}
    for name in models:
        P0, P1 = cross_validate_pair(name, bp, D["m2"], D["pid"], idx, clear, N_FOLDS)
        evs = {}
        for tag, P in (("E0", P0), ("E1", P1)):
            e = event_aggregate(D["rec"][idx], D["eid"][idx], D["pid"][idx], D["m2"][idx], P)
            # clarté de l'événement = celle de ses fenêtres (identique par construction)
            cl = pd.DataFrame(dict(rec=D["rec"][idx], eid=D["eid"][idx], clear=clear[idx])).groupby(["rec", "eid"]).clear.first().reset_index()
            e = e.merge(cl, on=["rec", "eid"])
            evs[tag] = e
        e0, e1 = evs["E0"], evs["E1"]
        lines += ["", f"=== modèle {name} ==="]
        # 1) part de cas limites par classe
        known = e0[e0.clear >= 0]
        part = []
        for c in range(3):
            s = known[known.y == c]
            part.append(f"{NAMES[c]} {int((s.clear == 0).sum())}/{len(s)} ({(s.clear == 0).mean():.0%})" if len(s) else f"{NAMES[c]} -")
        lines.append("cas LIMITES / événements, par classe de label : " + " | ".join(part))
        lines.append("E0 = entraînement sur tous les cas ; événements évalués avec les labels d'origine :")
        for sub, tag in ((None, "tous les événements"), (1, "cas clairs seulement"), (0, "cas limites seulement")):
            s = e0 if sub is None else e0[e0.clear == sub]
            lines.append(fmt(confusion_by_patient(s.pid, s.y, s.pred, patients), bi, "E0 " + tag))
        lines.append("E1 = entraînement sur les cas CLAIRS seulement, évaluation sur tous :")
        for sub, tag in ((None, "tous les événements"), (1, "cas clairs seulement"), (0, "cas limites seulement")):
            s = e1 if sub is None else e1[e1.clear == sub]
            lines.append(fmt(confusion_by_patient(s.pid, s.y, s.pred, patients), bi, "E1 " + tag))
        C0 = confusion_by_patient(e0.pid, e0.y, e0.pred, patients)
        C1 = confusion_by_patient(e1.pid, e1.y, e1.pred, patients)
        d = boot_bacc(C1, bi) - boot_bacc(C0, bi)
        lo, hi = ci(d)
        dm = bacc_acc(C1)[0] - bacc_acc(C0)[0]
        verdict = "E1 meilleur (IC exclut 0)" if lo > 0 else ("E1 moins bon (IC exclut 0)" if hi < 0 else "pas de différence détectable (IC contient 0)")
        lines.append(f"différence appariée E1 - E0 (équilibrée, tous événements) : {dm:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
        # 3) incertain : on garde les événements les plus confiants de E0
        conf = e0[["pL", "pR", "pB"]].values.max(1)
        lines.append("« Incertain » (E0) : on décide seulement les événements où le modèle est le plus confiant :")
        lines.append("  couverture | équilibrée [IC95]        | exactitude | part de cas limites parmi les décidés")
        for q in COVERAGES:
            keep = conf >= np.quantile(conf, 1 - q) if q < 1 else np.ones(len(e0), bool)
            s = e0[keep]
            Ck = confusion_by_patient(s.pid, s.y, s.pred, patients)
            b, a, _ = bacc_acc(Ck)
            l2, h2 = ci(boot_bacc(Ck, bi))
            kn = s[s.clear >= 0]
            lines.append(f"  {q:>9.0%} | {b:.3f} [{l2:.3f}, {h2:.3f}] | {a:.3f}      | {(kn.clear == 0).mean():.0%}")
        results[name] = dict(E0=e0, E1=e1, diff=(dm, lo, hi))
    return lines, results


def main(args):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    table = clarity_table(args.events_csv)
    D = load_cache(args.cache_dir, args.split_csv, roles=("train", "val"), limit_per_role=args.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    lines, res = analyse(D, bp, table, [m for m in args.models.split(",") if m])
    os.makedirs(args.out_dir, exist_ok=True)
    for name, r in res.items():
        r["E0"].to_csv(os.path.join(args.out_dir, f"{name}_E0_events.csv"), index=False)
        r["E1"].to_csv(os.path.join(args.out_dir, f"{name}_E1_events.csv"), index=False)
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

    # 1) définition de la clarté
    check("(4,0) sans foyer : gauche pour toutes les règles -> claire", is_clear(False, 4, 0) is True)
    check("(0,4) sans foyer : droite -> claire", is_clear(False, 0, 4) is True)
    check("(4,4) : bilatéral pour toutes les règles -> claire", is_clear(False, 4, 4) is True)
    check("(2,1) sans foyer : le label change avec rho -> limite", is_clear(False, 2, 1) is False)
    check("(0,0) : sans label -> None", is_clear(False, 0, 0) is None)
    # 2) bootstrap : deux expériences identiques -> différence nulle
    rng = np.random.default_rng(0)
    pid = rng.integers(0, 12, 300)
    y = rng.integers(0, 3, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, 300))
    pats = sorted(set(pid))
    C = confusion_by_patient(pid, y, pred, pats)
    bi = boot_idx(len(pats), 300)
    d = boot_bacc(C, bi) - boot_bacc(C, bi)
    check("bootstrap apparié : différence entre deux expériences identiques = 0 exactement", np.allclose(d, 0))
    b, a, r = bacc_acc(C)
    sk = float(np.mean([(pred[y == c] == c).mean() for c in range(3)]))
    check(f"exactitude équilibrée = moyenne des rappels ({b:.3f} = {sk:.3f})", abs(b - sk) < 1e-9)
    # 3) jeu synthétique : 30 patients x 3 enregistrements ; 1 enregistrement sur 5 = cas limite (aucune asymétrie, label aléatoire)
    n_pat, per_rec, nrec_pp = 30, 4, 3
    n_rec = n_pat * nrec_pp
    n = n_rec * per_rec
    rec = np.repeat(np.arange(n_rec), per_rec)
    patient_of_rec = np.arange(n_rec) // nrec_pp
    border_rec = (np.arange(n_rec) % 5 == 0)
    cls_rec = np.where(border_rec, rng.integers(0, 3, n_rec), np.arange(n_rec) % 3)
    cls = cls_rec[rec]
    x = rng.standard_normal((n, 18, 1000)).astype(np.float32) * 10
    for c, cols in ((0, range(0, 8)), (1, range(8, 16)), (2, range(0, 16))):
        m = (cls == c) & ~border_rec[rec]
        x[np.ix_(m, list(cols))] *= 3
    role = np.where(patient_of_rec[rec] < 20, "train", "val")
    D = dict(x=x.astype(np.float16), m2=cls.astype(np.int64), y=np.ones(n, np.int64), pid=patient_of_rec[rec], rec=rec, eid=np.zeros(n, np.int64),
             valid=np.tile(np.ones(18, bool), (n, 1)), role=role, rec_names=np.array([f"rec{i:03d}" for i in range(n_rec)]),
             patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.arange(n))
    nL = np.where(border_rec, 2, np.where(cls_rec == 0, 4, np.where(cls_rec == 1, 0, 4)))
    nR = np.where(border_rec, 1, np.where(cls_rec == 0, 0, 4))
    tmp = tempfile.mkdtemp()
    try:
        evp = os.path.join(tmp, "events.csv")
        pd.DataFrame(dict(recording=D["rec_names"], event_idx=0, has_focal=False, nL_channels=nL, nR_channels=nR)).to_csv(evp, index=False)
        table = clarity_table(evp)
        wc = window_clarity(D, table)
        check("clarté des fenêtres : les enregistrements 'limite' sont repérés (0) et les autres sont clairs (1), aucun inconnu",
              np.array_equal(wc, (~border_rec[rec]).astype(np.int8)))
        _, _, bp_all, _ = features_all(D)
        lines, res = analyse(D, bp_all, table, ["B_asym80"], B=200)
        e0, e1 = res["B_asym80"]["E0"], res["B_asym80"]["E1"]
        check(f"un événement par enregistrement ({len(e0)} = {n_rec})", len(e0) == n_rec and len(e1) == n_rec)
        acc_cl = (e0[e0.clear == 1].y == e0[e0.clear == 1].pred).mean()
        acc_bd = (e0[e0.clear == 0].y == e0[e0.clear == 0].pred).mean()
        check(f"cas clairs bien prédits ({acc_cl:.2f} > 0,9) et cas limites au hasard ({acc_bd:.2f} < 0,7)", acc_cl > 0.9 and acc_bd < 0.7)
        txt = "\n".join(lines)
        check("rapport complet : part de limites, E0, E1, différence appariée, courbe de couverture", all(s in txt for s in
              ("cas LIMITES", "E0 tous les événements", "E1 cas limites seulement", "différence appariée", "100%", "50%")))
        # couverture : exactitude des décidés au plus bas niveau >= exactitude pleine (les cas limites ont une probabilité floue)
        conf = e0[["pL", "pR", "pB"]].values.max(1)
        full = (e0.y == e0.pred).mean()
        top = e0[conf >= np.quantile(conf, 0.5)]
        check(f"incertain : exactitude des 50 % les plus confiants {(top.y == top.pred).mean():.2f} >= exactitude totale {full:.2f} - 0,02", (top.y == top.pred).mean() >= full - 0.02)
        # aucune fuite : plis par patient
        fo = patient_folds(D["pid"])
        check("plis par patient : aucun patient dans deux plis", all(len({fo[p]}) == 1 for p in fo))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--events_csv", default=base + "etape1/events.csv", help="events.csv écrit par l'étape 1 (colonnes nL_channels, nR_channels, has_focal)")
    ap.add_argument("--out_dir", default=base + "clarity")
    ap.add_argument("--models", default="B_asym80,A_lr90")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
