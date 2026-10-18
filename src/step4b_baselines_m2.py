#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 4b -- Repères simples pour M2 (sans réseau) : régression logistique sur la puissance de 5 bandes x 18 dérivations (90 variables),
avec miroir gauche/droite (variables permutées, labels gauche <-> droite échangés), poids égaux par (patient, classe).
Deux variantes : signal BRUT (garde l'amplitude et les rapports entre canaux) et signal NORMALISÉ par canal (comme l'ancien M1 : perd les rapports).
Évalué comme M2 : par fenêtre et par événement (probabilités moyennes), sur les patients de validation.

Usage :  python step4b_baselines_m2.py [--m2_scores .../m2_val_scores.csv ...]     |    --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_train_m1 import load_cache, FLIP_PERM
from step3b_baselines_m1 import features_all
from step4_train_m2 import m2_indices, m2_metrics, event_aggregate, FLIP_LABEL, fmt_metrics


def fit_lr_model(bp, y, pid, tr, flip=True):
    """Entraîne la régression logistique sur les fenêtres d'indices tr (avec miroir et poids égaux par (patient, classe)).
    Retourne (scaler, clf) : réutilisable sur de nouvelles fenêtres avec predict_lr_model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, ptr = bp[tr], y[tr], pid[tr]
    if flip:
        Xf = Xtr.reshape(len(Xtr), 18, -1)[:, FLIP_PERM, :].reshape(len(Xtr), -1)
        Xtr, ytr, ptr = np.concatenate([Xtr, Xf]), np.concatenate([ytr, FLIP_LABEL[ytr]]), np.concatenate([ptr, ptr])
    key = pd.Series(ptr).astype(str) + "_" + pd.Series(ytr).astype(str)
    w = 1.0 / key.map(key.value_counts()).values
    for c in range(3):
        w[ytr == c] /= w[ytr == c].sum()
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr, sample_weight=w * len(w))
    return sc, clf


def predict_lr_model(model, X):
    """Probabilités (n, 3) colonnes G/D/B ; une classe absente de l'entraînement reste à 0."""
    sc, clf = model
    P = np.zeros((len(X), 3))
    P[:, clf.classes_] = clf.predict_proba(sc.transform(X))
    return P


def fit_predict_lr(bp, y, pid, tr, va, flip=True):
    """bp : (n, 90) variables ; tr/va : indices. Retourne les probabilités (len(va), 3) de la régression logistique."""
    return predict_lr_model(fit_lr_model(bp, y, pid, tr, flip), bp[va])


def evaluate(D, itv, P):
    mw = m2_metrics(D["m2"][itv], P.argmax(1), D["pid"][itv])
    ev = event_aggregate(D["rec"][itv], D["eid"][itv], D["pid"][itv], D["m2"][itv], P)
    me = m2_metrics(ev.y.values, ev.pred.values, ev.pid.values)
    return mw, me, ev


def run(D, out_dir=None):
    itr, itv = m2_indices(D, "train"), m2_indices(D, "val")
    amp, ll, bp, bpn = features_all(D)
    out = {}
    for name, f, stem in (("rég. logistique, signal BRUT", bp, "lr_raw"), ("rég. logistique, signal NORMALISÉ par canal", bpn, "lr_norm")):
        P = fit_predict_lr(f, D["m2"], D["pid"], itr, itv)
        out[name] = evaluate(D, itv, P)[:2]
        if out_dir:                                   # mêmes colonnes que m2_val_scores.csv : comparables avec les réseaux
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(dict(recording=D["rec_names"][D["rec"][itv]], eid=D["eid"][itv], start_sample=D["start"][itv],
                              patient=D["patients"][D["pid"][itv]], y=D["m2"][itv], pL=P[:, 0], pR=P[:, 1], pB=P[:, 2])
                         ).to_csv(os.path.join(out_dir, f"{stem}_val_scores.csv"), index=False)
    return out


def main(args):
    D = load_cache(args.cache_dir, args.split_csv, limit_per_role=args.limit_per_role, verbose=False)
    itv = m2_indices(D, "val")
    ev0 = event_aggregate(D["rec"][itv], D["eid"][itv], D["pid"][itv], D["m2"][itv], np.full((len(itv), 3), 1 / 3))
    L = ["<<<COPY_FROM_HERE>>>", f"validation : {len(itv)} fenêtres, {len(ev0)} événements (G/D/B {np.bincount(ev0.y.values, minlength=3).tolist()}) | "
         f"taux de la classe majoritaire (événements) {max(np.bincount(ev0.y.values, minlength=3)) / len(ev0):.3f}"]
    for name, (mw, me) in run(D, args.out_dir).items():
        L.append(f"--- {name} ---")
        L.append(fmt_metrics(mw, "  fenêtres"))
        L.append(fmt_metrics(me, "  événements"))
    for path in (args.m2_scores or []):
        s = pd.read_csv(path)
        P = s[["pL", "pR", "pB"]].values
        ev = event_aggregate(s.recording.values, s.eid.values, s.patient.values, s.y.values, P)
        mw = m2_metrics(s.y.values, P.argmax(1), s.patient.values)
        me = m2_metrics(ev.y.values, ev.pred.values, ev.pid.values)
        L.append(f"--- M2 : {os.path.basename(os.path.dirname(os.path.abspath(path)))} ---")
        L.append(fmt_metrics(mw, "  fenêtres"))
        L.append(fmt_metrics(me, "  événements"))
    L.append("<<<COPY_TO_HERE>>>")
    print("\n".join(L))


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    n_rec, per = 18, 10
    n = n_rec * per
    x = rng.standard_normal((n, 18, 1000)).astype(np.float32) * 10
    cls = np.repeat(np.arange(n_rec) % 3, per)
    for c, cols in ((0, range(0, 8)), (1, range(8, 16)), (2, range(0, 16))):
        x[np.ix_(cls == c, list(cols))] *= 3                         # gauche : canaux 0-7 x3 ; droite : 8-15 x3 ; bilatéral : les deux
    rec = np.repeat(np.arange(n_rec), per)
    D = dict(x=x.astype(np.float16), m2=cls.astype(np.int64), y=np.ones(n, np.int64), pid=rec, rec=rec, eid=np.zeros(n, np.int64),
             valid=np.tile(np.ones(18, bool), (n, 1)), role=np.where(rec < 12, "train", "val"))
    import tempfile, shutil
    D.update(rec_names=np.array([f"rec{i:02d}" for i in range(n_rec)]), patients=np.array([f"p{i:02d}" for i in range(n_rec)]), start=np.arange(n))
    tmp = tempfile.mkdtemp()
    res = run(D, tmp)
    sc = pd.read_csv(os.path.join(tmp, "lr_raw_val_scores.csv"))
    check("scores LR écrits au format de m2_val_scores.csv (colonnes, une ligne par fenêtre de validation)",
          list(sc.columns) == ["recording", "eid", "start_sample", "patient", "y", "pL", "pR", "pB"] and len(sc) == int((D["role"] == "val").sum())
          and np.allclose(sc[["pL", "pR", "pB"]].sum(axis=1), 1.0))
    shutil.rmtree(tmp, ignore_errors=True)
    raw = res["rég. logistique, signal BRUT"][1]["acc"]
    norm = res["rég. logistique, signal NORMALISÉ par canal"][1]["acc"]
    check(f"asymétrie d'amplitude : signal brut, exactitude événements {raw:.2f} > 0,9", raw > 0.9)
    check(f"asymétrie d'amplitude : signal normalisé par canal {norm:.2f} < 0,7 (les rapports entre canaux sont perdus)", norm < 0.7)
    # le miroir : un modèle entraîné sur 'gauche' seulement prédit 'droite' pour des fenêtres miroir
    Xb = np.zeros((40, 90))
    Xb[:20] = rng.normal(0, 1, (20, 90)); Xb[:20].reshape(20, 18, 5)[:, 0:8, :] += 3
    Xb[20:] = rng.normal(0, 1, (20, 90)); Xb[20:].reshape(20, 18, 5)[:, 8:16, :] += 3
    yb = np.r_[np.zeros(20), np.ones(20)].astype(np.int64)
    P = fit_predict_lr(Xb, yb, np.arange(40), np.arange(0, 40, 2), np.arange(1, 40, 2), flip=True)
    check("fit_predict_lr : forme (n,3) et prédictions correctes sur un cas simple", P.shape == (20, 3) and (P.argmax(1) == yb[1::2]).mean() > 0.9)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--m2_scores", nargs="*", default=None, help="un ou plusieurs m2_val_scores.csv")
    ap.add_argument("--out_dir", default=None, help="dossier où écrire lr_raw_val_scores.csv et lr_norm_val_scores.csv")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
