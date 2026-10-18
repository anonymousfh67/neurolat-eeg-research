#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 12 -- Une idée de fond (pas un nouveau modèle) : mesurer la crise RELATIVEMENT au fond de LA MÊME enregistrement.  AUCUN accès à eval.

Problème. Jusqu'ici M2 reçoit la puissance brute (log) de chaque dérivation. Or une asymétrie gauche/droite peut exister AVANT la crise (électrodes, impédance,
anatomie) : le modèle confond « hémisphère plus fort au repos » et « hémisphère touché par la crise », surtout pour décider « bilatéral » (les deux côtés montent
par rapport à leur propre repos). Solution classique en EEG : soustraire le niveau de base de l'enregistrement.
Le cache contient jusqu'à 8 fenêtres de FOND (fenêtres sans aucune crise) par enregistrement : baseline = médiane de leur log-puissance, par dérivation et bande.
Les trois variantes sont décidées À L'AVANCE (aucun réglage) et TOUTES rapportées, en validation croisée PAR PATIENT (mêmes plis que l'étape 11) :

  F0_ref   : référence = B_asym80 sur la log-puissance brute (étape 7), agrégation par événement
  F1_ratio : même modèle, mais sur (log-puissance - baseline de l'enregistrement)
  F2_event : modèle PAR ÉVÉNEMENT sur le rapport crise/fond : asymétrie moyenne par paire (40), puis par bande : moyenne et maximum de |G-D|, maximum
             de chaque hémisphère, minimum des deux maxima (« les deux côtés montent »), moyenne G/D (30)

Une variante n'est dite meilleure que la référence que si l'IC95 (bootstrap apparié par patients) de la différence d'exactitude équilibrée par événement
exclut 0. Si le fond d'un enregistrement est absent, on prend celui du patient, puis le global (nombre signalé).

Usage :
  python step12_baseline_ratio.py [--out_dir .../ratio]
  python step12_baseline_ratio.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_train_m1 import FLIP_PERM
from step4_train_m2 import m2_indices, event_aggregate, FLIP_LABEL
from step7_features_m2_cv import patient_folds, fit_model, predict_model, mirror_bp, sample_weights, N_FOLDS
from step11_label_clarity import confusion_by_patient, boot_idx, boot_bacc, bacc_acc, ci, fmt

NB = 5
ORDER = ["F0_ref", "F1_ratio", "F2_event"]


# ------------------------------------------------------------------ baseline de l'enregistrement
def recording_baseline(D, bp):
    """(n_fenêtres, 90) : médiane de la log-puissance des fenêtres de fond (D['y'] == 0) du même enregistrement ; repli patient puis global.
    Retourne aussi le nombre d'enregistrements sans fond propre."""
    rec, pid, y = D["rec"], D["pid"], D["y"]
    n_rec = int(rec.max()) + 1
    base = np.full((n_rec, bp.shape[1]), np.nan, np.float32)
    for r in range(n_rec):
        m = (rec == r) & (y == 0)
        if m.any():
            base[r] = np.median(bp[m], axis=0)
    rec_pat = np.zeros(n_rec, int)
    rec_pat[rec] = pid
    n_missing = int(np.isnan(base[:, 0]).sum())
    glob = np.nanmedian(base, axis=0) if n_missing < n_rec else np.zeros(bp.shape[1], np.float32)
    for r in np.where(np.isnan(base[:, 0]))[0]:
        same = [q for q in range(n_rec) if rec_pat[q] == rec_pat[r] and not np.isnan(base[q, 0])]
        base[r] = np.median(base[same], axis=0) if same else glob
    return base[rec], n_missing


# ------------------------------------------------------------------ variables par événement
def event_features(rows):
    """rows : (n, 90) rapport crise/fond des fenêtres d'UN événement -> vecteur de 70 variables (voir en-tête)."""
    R = np.asarray(rows, np.float32).reshape(len(rows), 18, NB)
    Lc, Rc = R[:, :8], R[:, 8:16]
    d = (Lc - Rc)                                             # (n, 8, 5), change de signe au miroir
    L, Rr = Lc.mean(1), Rc.mean(1)                            # (n, 5)
    dh = L - Rr
    maxL, maxR = L.max(0), Rr.max(0)
    return np.concatenate([d.mean(0).reshape(-1), np.abs(dh).mean(0), np.abs(dh).max(0), maxL, maxR, np.minimum(maxL, maxR), ((L + Rr) / 2).mean(0)])


def build_events(idx, rec, eid, y, pid, R):
    """Regroupe les fenêtres idx par (enregistrement, événement). Retourne dict(keys, y, pid, windows=[indices])."""
    df = pd.DataFrame(dict(i=idx, rec=rec[idx], eid=eid[idx]))
    keys, wins = [], []
    for (r, e), g in df.groupby(["rec", "eid"], sort=True):
        keys.append((r, e))
        wins.append(g.i.values)
    ev_y = np.array([y[w[0]] for w in wins])
    ev_pid = np.array([pid[w[0]] for w in wins])
    return dict(keys=keys, y=ev_y, pid=ev_pid, wins=wins)


def fit_event_model(E, R, tr_ev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, Y, P = [], [], []
    for k in tr_ev:
        w = E["wins"][k]
        X.append(event_features(R[w])); Y.append(E["y"][k]); P.append(E["pid"][k])
        X.append(event_features(mirror_bp(R[w]))); Y.append(FLIP_LABEL[E["y"][k]]); P.append(E["pid"][k])
    X, Y, P = np.array(X), np.array(Y), np.array(P)
    wt = sample_weights(Y, P)
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), Y, sample_weight=wt)


def predict_event_model(model, E, R, te_ev):
    sc, clf = model
    P = np.zeros((len(te_ev), 3))
    X = np.array([event_features(R[E["wins"][k]]) for k in te_ev])
    P[:, clf.classes_] = clf.predict_proba(sc.transform(X))
    return P


# ------------------------------------------------------------------ validation croisée
def run_cv(D, bp, B=2000):
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx.sort()
    base, n_missing = recording_baseline(D, bp)
    R = (bp - base).astype(np.float32)
    fold_of = patient_folds(D["pid"][idx], N_FOLDS)
    fold = np.array([fold_of[p] for p in D["pid"][idx]])
    P0, P1 = np.zeros((len(idx), 3)), np.zeros((len(idx), 3))
    E = build_events(idx, D["rec"], D["eid"], D["m2"], D["pid"], R)
    ev_fold = np.array([fold_of[p] for p in E["pid"]])
    P2 = np.zeros((len(E["y"]), 3))
    for k in range(N_FOLDS):
        tr, te = idx[fold != k], idx[fold == k]
        if len(te) == 0:
            continue
        P0[fold == k] = predict_model(fit_model("B_asym80", bp, D["m2"], D["pid"], tr), bp[te])
        P1[fold == k] = predict_model(fit_model("B_asym80", R, D["m2"], D["pid"], tr), R[te])
        te_ev = np.where(ev_fold == k)[0]
        P2[te_ev] = predict_event_model(fit_event_model(E, R, np.where(ev_fold != k)[0]), E, R, te_ev)
    evs = {}
    for name, P in (("F0_ref", P0), ("F1_ratio", P1)):
        e = event_aggregate(D["rec"][idx], D["eid"][idx], D["pid"][idx], D["m2"][idx], P)
        evs[name] = e.set_index(["rec", "eid"])
    e2 = pd.DataFrame(dict(rec=[k[0] for k in E["keys"]], eid=[k[1] for k in E["keys"]], y=E["y"], pid=E["pid"],
                           pL=P2[:, 0], pR=P2[:, 1], pB=P2[:, 2]))
    e2["pred"] = P2.argmax(1)
    evs["F2_event"] = e2.set_index(["rec", "eid"])
    common = evs["F0_ref"].index
    for n in ORDER:
        evs[n] = evs[n].loc[common].reset_index()
    patients = sorted(set(D["pid"][idx]))
    bi = boot_idx(len(patients), B)
    lines = [f"{len(idx)} fenêtres M2, {len(common)} événements, {len(patients)} patients (train + val), {N_FOLDS} plis par patient ; "
             f"enregistrements sans fond propre (repli patient/global) : {n_missing}", "Évaluation avec les labels d'origine, par événement :"]
    C = {n: confusion_by_patient(evs[n].pid, evs[n].y, evs[n].pred, patients) for n in ORDER}
    for n in ORDER:
        lines.append(fmt(C[n], bi, n))
    b0 = boot_bacc(C["F0_ref"], bi)
    res = {}
    for n in ORDER[1:]:
        d = boot_bacc(C[n], bi) - b0
        lo, hi = ci(d)
        dm = bacc_acc(C[n])[0] - bacc_acc(C["F0_ref"])[0]
        v = f"{n} MEILLEUR (IC exclut 0)" if lo > 0 else (f"{n} moins bon (IC exclut 0)" if hi < 0 else "pas de différence détectable (IC contient 0)")
        lines.append(f"différence appariée {n} - F0_ref (équilibrée) : {dm:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}")
        res[n] = (dm, lo, hi)
    return lines, evs, res


def main(args):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    D = load_cache(args.cache_dir, args.split_csv, roles=("train", "val"), limit_per_role=args.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    lines, evs, _ = run_cv(D, bp)
    os.makedirs(args.out_dir, exist_ok=True)
    for n, e in evs.items():
        e.to_csv(os.path.join(args.out_dir, f"{n}_events.csv"), index=False)
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
    # 1) baseline : médiane des fenêtres de fond du même enregistrement ; repli patient ; miroir cohérent
    n = 20
    D0 = dict(rec=np.repeat([0, 1, 2], [8, 6, 6]), pid=np.repeat([0, 0, 1], [8, 6, 6]), y=np.r_[np.zeros(8), np.zeros(3), np.ones(3), np.ones(6)].astype(int))
    bp0 = rng.normal(0, 1, (n, 90)).astype(np.float32)
    base, miss = recording_baseline(D0, bp0)
    check("baseline = médiane des fenêtres de fond du même enregistrement", np.allclose(base[0], np.median(bp0[:8], axis=0)) and np.allclose(base[9], np.median(bp0[8:11], axis=0)))
    check("enregistrement sans fond : repli sur le global (1 manquant signalé)", miss == 1 and np.all(np.isfinite(base[-1])))
    check("miroir cohérent : (bp - fond) miroir = miroir(bp) - miroir(fond)", np.allclose(mirror_bp(bp0 - base), mirror_bp(bp0) - mirror_bp(base), atol=1e-5))
    f = event_features(bp0[:5])
    fm = event_features(mirror_bp(bp0[:5]))
    check("70 variables par événement ; asymétrie moyenne par paire (40) change de signe au miroir", len(f) == 70 and np.allclose(f[:40], -fm[:40], atol=1e-5))
    check("variables symétriques (|G-D|, min des maxima, moyenne G/D) inchangées par le miroir", np.allclose(f[40:50], fm[40:50], atol=1e-5) and np.allclose(f[65:], fm[65:], atol=1e-5))
    # 2) jeu synthétique : asymétrie de REPOS aléatoire (confondante) + effet de crise ; la baseline doit la supprimer
    n_pat, nrec_pp, bg, ev_w = 30, 3, 8, 4
    n_rec = n_pat * nrec_pp
    per = bg + ev_w
    rec = np.repeat(np.arange(n_rec), per)
    pat_rec = np.arange(n_rec) // nrec_pp
    cls_rec = np.arange(n_rec) % 3
    gL = np.exp(rng.uniform(-0.9, 0.9, n_rec)); gR = np.exp(rng.uniform(-0.9, 0.9, n_rec))
    x = rng.standard_normal((n_rec * per, 18, 1000)).astype(np.float32) * 10
    y = np.zeros(n_rec * per, int); m2 = np.full(n_rec * per, -1); eid = np.full(n_rec * per, -1)
    for r in range(n_rec):
        sl = slice(r * per, (r + 1) * per)
        x[sl, :8] *= gL[r]; x[sl, 8:16] *= gR[r]                      # asymétrie de repos propre à l'enregistrement
        sz = slice(r * per + bg, (r + 1) * per)
        c = cls_rec[r]
        cols = {0: range(0, 8), 1: range(8, 16), 2: range(0, 16)}[c]
        x[sz][:, list(cols)] *= 3.0
        y[sz] = 1; m2[sz] = c; eid[sz] = 0
    role = np.where(pat_rec[rec] < 20, "train", "val")
    D = dict(x=x.astype(np.float16), m2=m2, y=y, pid=pat_rec[rec], rec=rec, eid=eid, valid=np.tile(np.ones(18, bool), (len(rec), 1)), role=role,
             rec_names=np.array([f"rec{i:03d}" for i in range(n_rec)]), patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.arange(len(rec)))
    _, _, bp, _ = features_all(D)
    lines, evs, res = run_cv(D, bp, B=200)
    acc = {n: float((evs[n].y == evs[n].pred).mean()) for n in ORDER}
    check(f"un événement par enregistrement ({len(evs['F0_ref'])} = {n_rec}), mêmes événements dans les 3 variantes", all(len(evs[n]) == n_rec for n in ORDER))
    check(f"avec asymétrie de repos confondante, la baseline aide : exactitude F1 {acc['F1_ratio']:.2f} et F2 {acc['F2_event']:.2f} > F0 {acc['F0_ref']:.2f} + 0,10",
          acc["F1_ratio"] > acc["F0_ref"] + 0.10 and acc["F2_event"] > acc["F0_ref"] + 0.10)
    check("F1 et F2 près de la perfection sur cet exemple (> 0,9)", acc["F1_ratio"] > 0.9 and acc["F2_event"] > 0.9)
    check("rapport : trois lignes de variantes et deux différences appariées", sum("différence appariée" in l for l in lines) == 2)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--out_dir", default=base + "ratio")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
