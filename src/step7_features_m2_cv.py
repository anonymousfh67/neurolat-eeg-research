#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 7 -- Chercher un meilleur M2 SANS toucher à eval : variables d'ASYMÉTRIE gauche/droite + validation croisée PAR PATIENT.

Constat des étapes précédentes : l'information utile pour gauche / droite / bilatéral est l'asymétrie d'amplitude entre les deux hémisphères ;
une régression logistique sur 90 variables (5 bandes x 18 dérivations) fait aussi bien que le réseau. Ici on donne cette asymétrie directement au modèle.
Toutes les variantes sont décidées À L'AVANCE (aucun réglage d'hyperparamètre) et TOUTES sont rapportées :

  A_lr90   : référence = régression logistique sur les 90 variables brutes (comme étape 4b)
  B_asym80 : régression logistique ; par bande et par paire miroir (8 paires) : différence G-D (40, change de signe au miroir)
             et moyenne G/D (40, ne change pas)
  C_compact: régression logistique ; par bande : moyenne de la différence G-D, moyenne de |G-D|, maximum de |G-D|, puissance moyenne,
             + puissance des 2 dérivations médianes (30 variables)
  D_hgb_B  : arbres de gradient (profondeur 3, 150 itérations, paramètres fixés) sur les variables de B

Protocole : patients des rôles train et val réunis, 5 plis PAR PATIENT (un patient n'est jamais à la fois dans l'entraînement et dans le test d'un pli),
miroir gauche/droite en entraînement, poids égaux par (patient, classe). Chaque fenêtre reçoit une prédiction « hors pli » ; les scores sont écrits au
format de l'étape 4b pour être comparés avec step4c_compare_m2.py (bootstrap par patient, différences appariées).

Usage :
  python step7_features_m2_cv.py [--out_dir .../lr_cv]       puis
  python step4c_compare_m2.py --ref .../lr_cv/A_lr90_val_scores.csv .../lr_cv/B_asym80_val_scores.csv ...
  python step7_features_m2_cv.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_train_m1 import FLIP_PERM
from step4_train_m2 import m2_indices, m2_metrics, event_aggregate, FLIP_LABEL, fmt_metrics

N_FOLDS = 5
NB = 5          # nombre de bandes
ORDER = ["A_lr90", "B_asym80", "C_compact", "D_hgb_B"]


# ------------------------------------------------------------------ variables
def _bp3(bp):
    return np.asarray(bp, np.float32).reshape(len(bp), 18, NB)


def feat_A(bp):
    return np.asarray(bp, np.float32).reshape(len(bp), -1)


def feat_B(bp):
    b = _bp3(bp)
    left, right = b[:, :8], b[:, 8:16]            # canaux 0-7 = hémisphère gauche, 8-15 = miroirs droits (FLIP_PERM échange 0-7 <-> 8-15)
    return np.concatenate([(left - right).reshape(len(b), -1), ((left + right) / 2).reshape(len(b), -1)], axis=1)


def feat_C(bp):
    b = _bp3(bp)
    d = b[:, :8] - b[:, 8:16]                      # (n, 8, 5)
    mid = b[:, 16:18].reshape(len(b), -1)
    return np.concatenate([d.mean(1), np.abs(d).mean(1), np.abs(d).max(1), b[:, :16].mean(1), mid], axis=1)


FEATURES = {"A_lr90": feat_A, "B_asym80": feat_B, "C_compact": feat_C, "D_hgb_B": feat_B}


def mirror_bp(bp):
    """Variables d'une fenêtre dont les deux hémisphères sont échangés (même opération que FLIP_PERM sur les canaux)."""
    return _bp3(bp)[:, FLIP_PERM, :].reshape(len(bp), -1)


# ------------------------------------------------------------------ modèles
def sample_weights(y, pid):
    key = pd.Series(pid).astype(str) + "_" + pd.Series(y).astype(str)
    w = 1.0 / key.map(key.value_counts()).values
    for c in range(3):
        if (y == c).any():
            w[y == c] /= w[y == c].sum()
    return w * len(w)


def fit_model(name, bp, y, pid, tr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    f = FEATURES[name]
    Xtr = np.concatenate([f(bp[tr]), f(mirror_bp(bp[tr]))])               # miroir : variables recalculées sur la fenêtre échangée
    ytr = np.concatenate([y[tr], FLIP_LABEL[y[tr]]])
    ptr = np.concatenate([pid[tr], pid[tr]])
    w = sample_weights(ytr, ptr)
    if name == "D_hgb_B":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=0)
        clf.fit(Xtr, ytr, sample_weight=w)
        sc = None
    else:
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr, sample_weight=w)
    return name, sc, clf


def predict_model(model, bp):
    name, sc, clf = model
    X = FEATURES[name](bp)
    P = np.zeros((len(bp), 3))
    P[:, clf.classes_] = clf.predict_proba(X if sc is None else sc.transform(X))
    return P


# ------------------------------------------------------------------ validation croisée par patient
def patient_folds(pid_values, n_folds=N_FOLDS, seed=0):
    """Affecte chaque patient à un pli (équilibré en nombre de patients) : dict patient -> pli."""
    pats = np.array(sorted(set(pid_values)))
    perm = np.random.default_rng(seed).permutation(len(pats))
    return {pats[i]: k % n_folds for k, i in enumerate(perm)}


def cross_validate(bp, y, pid, idx, names=ORDER, n_folds=N_FOLDS):
    """bp : (n, 90) toutes fenêtres ; idx : fenêtres M2 utilisées ; retourne {nom: probabilités hors pli (len(idx), 3)} et le pli de chaque fenêtre."""
    fold_of = patient_folds(pid[idx], n_folds)
    fold = np.array([fold_of[p] for p in pid[idx]])
    out = {n: np.zeros((len(idx), 3)) for n in names}
    for k in range(n_folds):
        tr, te = idx[fold != k], idx[fold == k]
        if len(te) == 0:
            continue
        for n in names:
            out[n][fold == k] = predict_model(fit_model(n, bp, y, pid, tr), bp[te])
    return out, fold


def write_scores(D, idx, P, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(dict(recording=D["rec_names"][D["rec"][idx]], eid=D["eid"][idx], start_sample=D["start"][idx], patient=D["patients"][D["pid"][idx]],
                      y=D["m2"][idx], pL=P[:, 0], pR=P[:, 1], pB=P[:, 2])).to_csv(path, index=False)


def run(D, bp, out_dir=None):
    idx = np.concatenate([m2_indices(D, "train"), m2_indices(D, "val")])
    idx.sort()
    P, fold = cross_validate(bp, D["m2"], D["pid"], idx)
    rows = {}
    for n in ORDER:
        mw = m2_metrics(D["m2"][idx], P[n].argmax(1), D["pid"][idx])
        ev = event_aggregate(D["rec"][idx], D["eid"][idx], D["pid"][idx], D["m2"][idx], P[n])
        rows[n] = (mw, m2_metrics(ev.y.values, ev.pred.values, ev.pid.values))
        if out_dir:
            write_scores(D, idx, P[n], os.path.join(out_dir, f"{n}_val_scores.csv"))
    return rows, idx, fold


def main(args):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    D = load_cache(args.cache_dir, args.split_csv, roles=("train", "val"), limit_per_role=args.limit_per_role, verbose=False)
    _, _, bp, _ = features_all(D)
    rows, idx, fold = run(D, bp, args.out_dir)
    L = ["<<<COPY_FROM_HERE>>>",
         f"{len(idx)} fenêtres M2, {len(set(D['pid'][idx]))} patients (train + val), {N_FOLDS} plis par patient ; scores hors pli écrits dans {args.out_dir}"]
    for n in ORDER:
        mw, me = rows[n]
        L.append(f"--- {n} ---")
        L.append(fmt_metrics(mw, "  fenêtres"))
        L.append(fmt_metrics(me, "  événements"))
    L.append("<<<COPY_TO_HERE>>>")
    print("\n".join(L))


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
    # 1) variables : le miroir change le signe de la différence G-D et laisse la moyenne G/D inchangée
    bp = rng.normal(0, 1, (6, 90))
    B, Bm = feat_B(bp), feat_B(mirror_bp(bp))
    check("feat_B : différence G-D change de signe au miroir, moyenne G/D inchangée",
          np.allclose(B[:, :40], -Bm[:, :40], atol=1e-5) and np.allclose(B[:, 40:], Bm[:, 40:], atol=1e-5))
    C, Cm = feat_C(bp), feat_C(mirror_bp(bp))
    check("feat_C : |G-D| et puissance moyenne inchangés par le miroir (30 variables)", C.shape[1] == 30 and np.allclose(C[:, 5:20], Cm[:, 5:20], atol=1e-5))
    check("miroir appliqué deux fois = identité", np.allclose(mirror_bp(mirror_bp(bp)), bp))
    # 2) plis : un patient dans un seul pli, tous les plis non vides
    pats = np.repeat(np.arange(23), 4)
    f = patient_folds(pats)
    check("plis : chaque patient dans exactement un pli, 5 plis non vides, tailles équilibrées (4-5 patients)",
          set(f.values()) == set(range(5)) and sorted(np.bincount(list(f.values()))) == [4, 4, 5, 5, 5])
    # 3) validation croisée sur un jeu synthétique : asymétrie d'amplitude gauche / droite / bilatérale, 30 patients
    n_pat, per = 30, 12
    n = n_pat * per
    cls = np.repeat(np.arange(n_pat) % 3, per)
    x = rng.standard_normal((n, 18, 1000)).astype(np.float32) * 10
    for c, cols in ((0, range(0, 8)), (1, range(8, 16)), (2, range(0, 16))):
        x[np.ix_(cls == c, list(cols))] *= 3
    rec = np.repeat(np.arange(n_pat), per)
    role = np.where(rec < 20, "train", "val")
    D = dict(x=x.astype(np.float16), m2=cls.astype(np.int64), y=np.ones(n, np.int64), pid=rec, rec=rec, eid=np.zeros(n, np.int64),
             valid=np.tile(np.ones(18, bool), (n, 1)), role=role, rec_names=np.array([f"rec{i:02d}" for i in range(n_pat)]),
             patients=np.array([f"p{i:02d}" for i in range(n_pat)]), start=np.arange(n))
    _, _, bp_all, _ = features_all(D)
    tmp = tempfile.mkdtemp()
    try:
        rows, idx, fold = run(D, bp_all, tmp)
        check(f"CV : toutes les fenêtres M2 (train + val) prédites une fois ({len(idx)} = {n})", len(idx) == n)
        for nme in ORDER:
            check(f"  {nme} : exactitude par événement {rows[nme][1]['acc']:.2f} (asymétrie nette -> attendue > 0,9)", rows[nme][1]["acc"] > 0.9)
        s = pd.read_csv(os.path.join(tmp, "B_asym80_val_scores.csv"))
        check("scores écrits au format de l'étape 4b (colonnes, une ligne par fenêtre, probabilités de somme 1)",
              list(s.columns) == ["recording", "eid", "start_sample", "patient", "y", "pL", "pR", "pB"] and len(s) == n and np.allclose(s[["pL", "pR", "pB"]].sum(axis=1), 1))
        # aucun patient d'un pli de test dans l'entraînement de ce pli : vérification directe
        fo = patient_folds(D["pid"][idx])
        leak = False
        for k in range(N_FOLDS):
            tr_p = {p for p in D["pid"][idx] if fo[p] != k}
            te_p = {p for p in D["pid"][idx] if fo[p] == k}
            leak |= bool(tr_p & te_p)
        check("aucune fuite de patient entre entraînement et test d'un pli", not leak)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--out_dir", default=base + "lr_cv")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
