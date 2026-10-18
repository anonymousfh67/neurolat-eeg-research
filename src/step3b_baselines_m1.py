#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 3b -- Repères simples pour M1 (sans réseau de neurones) + diagnostic des scores de M1.

Pourquoi : M1 obtient une AUC de validation de ~0,74. Est-ce beaucoup ou peu ? On compare avec trois méthodes simples,
entraînées / évaluées sur LES MÊMES patients (train -> val) et avec la même métrique (AUC, simple et pondérée par patient) :
  1. amplitude de la fenêtre (écart-type médian des dérivations, en µV, valeurs BRUTES)
  2. « line length » (somme des variations d'un échantillon au suivant), idem
  3. régression logistique sur la puissance de 5 bandes (0,5-4, 4-8, 8-13, 13-30, 30-70 Hz) x 18 dérivations = 90 variables
Puis, si --scores_csv est donné (m1_val_scores.csv) :
  4. AUC de M1 par patient (distribution) et tableau sensibilité / spécificité selon le seuil.

Usage :
    python step3b_baselines_m1.py --scores_csv .../m1_seed0/m1_val_scores.csv
    python step3b_baselines_m1.py --selftest
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_train_m1 import load_cache, val_metrics

FS = 250
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 70)]


def window_features(x, valid, normalize=False):
    """x : (n, 18, T) en µV (float16 ou 32), valid : (18,) -> amplitude (n,), line length (n,), bandes (n, 90).
    normalize=True : applique d'abord la normalisation de M1 (par canal et par fenêtre : centrage, division par l'écart-type, clip +-10)."""
    from scipy.signal import welch
    xf = x.astype(np.float32)[:, valid, :]
    if normalize:
        xf = xf - xf.mean(axis=2, keepdims=True)
        xf = np.clip(xf / (np.sqrt((xf ** 2).mean(axis=2, keepdims=True)) + 1e-8), -10, 10)
    n = len(xf)
    amp = np.median(xf.std(axis=2), axis=1)
    ll = np.median(np.abs(np.diff(xf, axis=2)).mean(axis=2), axis=1)
    f, P = welch(xf, fs=FS, nperseg=250, axis=-1)
    bp = np.stack([np.log10(P[:, :, (f >= a) & (f < b)].mean(-1) + 1e-12) for a, b in BANDS], axis=2)   # (n, nvalid, 5)
    full = np.full((n, 18, len(BANDS)), np.nan, np.float32)
    full[:, valid] = bp
    full = np.where(np.isnan(full), bp.mean(axis=1, keepdims=True), full)      # dérivation invalide -> moyenne des valides
    return amp, ll, full.reshape(n, -1)


def features_all(D, chunk=400):
    """Variables par enregistrement (fenêtres contiguës, même masque valid). Retourne amplitude, line length, bandes (signal brut)
    et bandes calculées sur le signal NORMALISÉ comme pour M1."""
    rec = D["rec"]
    bounds = np.r_[0, np.flatnonzero(np.diff(rec)) + 1, len(rec)]
    amp, ll = np.zeros(len(rec), np.float32), np.zeros(len(rec), np.float32)
    bp = np.zeros((len(rec), 18 * len(BANDS)), np.float32)
    bpn = np.zeros_like(bp)
    for a, b in zip(bounds[:-1], bounds[1:]):
        for i in range(a, b, chunk):
            j = min(i + chunk, b)
            amp[i:j], ll[i:j], bp[i:j] = window_features(D["x"][i:j], D["valid"][i])
            bpn[i:j] = window_features(D["x"][i:j], D["valid"][i], normalize=True)[2]
    return amp, ll, bp, bpn


def fit_lr(bp, D, tr, va):
    """Régression logistique : chaque (patient, classe) pèse pareil, les deux classes ont le même poids total."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    key = pd.Series(D["pid"][tr]).astype(str) + "_" + pd.Series(D["y"][tr]).astype(str)
    w = 1.0 / key.map(key.value_counts()).values
    for c in (0, 1):
        w[D["y"][tr] == c] /= w[D["y"][tr] == c].sum()
    sc = StandardScaler().fit(bp[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(bp[tr]), D["y"][tr], sample_weight=w * len(w))
    p = clf.predict_proba(sc.transform(bp[va]))[:, 1]
    m = val_metrics(D["y"][va], p, D["pid"][va])
    return m["auc"], m["auc_pw"], p


def run_baselines(D):
    tr, va = D["role"] == "train", D["role"] == "val"
    amp, ll, bp, bpn = features_all(D)
    out = {}
    for name, f in (("amplitude (µV)", amp), ("line length", ll)):
        m = val_metrics(D["y"][va], f[va], D["pid"][va])
        out[name] = (m["auc"], m["auc_pw"], f[va])
    out["rég. logistique, 5 bandes x 18, signal BRUT"] = fit_lr(bp, D, tr, va)
    out["rég. logistique, 5 bandes x 18, signal NORMALISÉ (comme M1)"] = fit_lr(bpn, D, tr, va)
    return out


def within_recording_auc(y, p, rec, pid, min_each=5):
    """AUC crise contre fond calculée DANS chaque enregistrement (>= min_each fenêtres de chaque classe), puis résumée.
    Elle est insensible aux différences d'amplitude ou de matériel ENTRE enregistrements."""
    from sklearn.metrics import roc_auc_score
    df = pd.DataFrame(dict(y=np.asarray(y), p=np.asarray(p, float), rec=np.asarray(rec), pid=np.asarray(pid)))
    rows = []
    for r, g in df.groupby("rec"):
        if (g.y == 1).sum() >= min_each and (g.y == 0).sum() >= min_each:
            rows.append(dict(rec=r, pid=g.pid.iloc[0], auc=roc_auc_score(g.y, g.p)))
    t = pd.DataFrame(rows)
    if t.empty:
        return None
    return dict(n_rec=len(t), median=float(t.auc.median()), mean_pat=float(t.groupby("pid").auc.mean().mean()), n_pat=int(t.pid.nunique()))


def fmt_within(w):
    return "n/a" if w is None else f"médiane {w['median']:.3f} ; moyenne par patient {w['mean_pat']:.3f} ({w['n_rec']} enreg., {w['n_pat']} patients)"


def diagnose_scores(scores):
    from sklearn.metrics import roc_auc_score
    L = []
    rows = []
    for pat, g in scores.groupby("patient"):
        if (g.y == 1).sum() >= 10 and (g.y == 0).sum() >= 10:
            rows.append(dict(patient=pat, auc=roc_auc_score(g.y, g.p), n=len(g)))
    t = pd.DataFrame(rows)
    if len(t):
        L.append(f"AUC de M1 PAR PATIENT (patients avec >= 10 fenêtres de chaque classe : {len(t)} sur {scores.patient.nunique()}) : "
                 f"médiane {t.auc.median():.3f}, quartiles {t.auc.quantile(.25):.3f}-{t.auc.quantile(.75):.3f}, "
                 f"patients < 0,5 : {int((t.auc < .5).sum())}, > 0,8 : {int((t.auc > .8).sum())}")
    else:
        L.append("AUC par patient : aucun patient n'a assez de fenêtres des deux classes")
    L.append("seuil | sensibilité (par patient) | spécificité (par patient)")
    for thr in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        m = val_metrics(scores.y.values, scores.p.values, scores.patient.values, thr=thr)
        L.append(f"{thr:.2f}  | {m['sens_pw']:.3f} | {m['spec_pw']:.3f}")
    return L


def main(args):
    L = ["<<<COPY_FROM_HERE>>>"]
    L.append("AUC de validation (mêmes patients) : AUC simple | AUC pondérée par patient | AUC INTRA-ENREGISTREMENT (crise vs fond du même enregistrement)")
    if not args.skip_baselines:
        D = load_cache(args.cache_dir, args.split_csv, limit_per_role=args.limit_per_role, verbose=True)
        va = D["role"] == "val"
        for k, (a, b, pv) in run_baselines(D).items():
            L.append(f"  {k:62s} {a:.3f} | {b:.3f} | {fmt_within(within_recording_auc(D['y'][va], pv, D['rec'][va], D['pid'][va]))}")
    extra = []
    for path in (args.scores_csv or []):
        s = pd.read_csv(path)
        name = "M1 : " + os.path.basename(os.path.dirname(os.path.abspath(path)))
        m = val_metrics(s.y.values, s.p.values, s.patient.values)
        L.append(f"  {name:62s} {m['auc']:.3f} | {m['auc_pw']:.3f} | {fmt_within(within_recording_auc(s.y, s.p, s.recording, s.patient))}")
        extra.append((name, s))
    for name, s in extra:
        L.append(f"--- {name} ---")
        L += diagnose_scores(s)
    L.append("<<<COPY_TO_HERE>>>")
    print("\n".join(L))


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n = 40
    t = np.arange(1000) / FS
    x = (rng.standard_normal((n, 18, 1000)) * 10).astype(np.float32)
    x[20:] += (30 * np.sin(2 * np.pi * 6 * t))[None, None, :]       # « crise » : rythme 6 Hz de forte amplitude
    valid = np.ones(18, bool)
    valid[3] = False
    x[:, 3] = 0
    amp, ll, bp = window_features(x.astype(np.float16), valid)
    check("window_features : formes (n,), (n,), (n,90)", amp.shape == (n,) and ll.shape == (n,) and bp.shape == (n, 90))
    check("amplitude plus forte pendant la « crise »", amp[20:].mean() > 2 * amp[:20].mean())
    check("pas de NaN malgré une dérivation invalide", np.isfinite(bp).all())
    y = np.r_[np.zeros(20), np.ones(20)]
    check("AUC de l'amplitude ~1 sur ce jeu synthétique", roc_auc_score(y, amp) > 0.95)
    # chaîne complète : chaque rôle contient les DEUX classes ; blocs de 10 fenêtres = 1 enregistrement = 1 patient
    D = dict(x=x.astype(np.float16), y=y.astype(np.int64), pid=np.repeat([0, 2, 1, 3], 10), rec=np.repeat(np.arange(4), 10),
             valid=np.tile(valid, (n, 1)),
             role=np.array(["train"] * 10 + ["val"] * 10 + ["train"] * 10 + ["val"] * 10))
    res = run_baselines(D)
    check(f"run_baselines : 3 méthodes, AUC val élevées ({ {k: round(v[0], 2) for k, v in res.items()} })",
          len(res) == 4 and all(v[0] > 0.9 for v in res.values()))
    # différence d'AMPLITUDE pure (même bruit x3) : le signal brut la voit, le signal normalisé ne doit plus la voir
    rng2 = np.random.default_rng(5)
    xa = rng2.standard_normal((200, 18, 1000)).astype(np.float32) * 10
    ya = np.tile(np.r_[np.zeros(25), np.ones(25)], 4).astype(np.int64)
    xa[ya == 1] *= 3
    Da = dict(x=xa.astype(np.float16), y=ya, pid=np.repeat(np.arange(4), 50), rec=np.repeat(np.arange(4), 50), valid=np.tile(np.ones(18, bool), (200, 1)),
              role=np.array(["train"] * 100 + ["val"] * 100))
    ra = run_baselines(Da)
    raw_auc = ra["rég. logistique, 5 bandes x 18, signal BRUT"][0]
    norm_auc = ra["rég. logistique, 5 bandes x 18, signal NORMALISÉ (comme M1)"][0]
    check(f"amplitude pure : AUC signal brut {raw_auc:.2f} > 0,95 ; signal normalisé {norm_auc:.2f} < 0,75", raw_auc > 0.95 and norm_auc < 0.75)
    # AUC intra-enregistrement : un décalage entre enregistrements fausse l'AUC globale mais pas l'AUC intra-enregistrement
    ys = np.tile(np.r_[np.ones(10), np.zeros(10)], 4)
    rec_ids = np.repeat(np.arange(4), 20)
    offset = np.array([0.0, 0.8, 0.0, 0.8])[rec_ids]                      # enregistrements 1 et 3 : tous les scores plus hauts
    ps = np.where(ys == 1, offset + 0.3, offset) + rng.normal(0, 0.05, 80)
    from sklearn.metrics import roc_auc_score as _auc
    wr = within_recording_auc(ys, ps, rec_ids, rec_ids // 2)
    check(f"AUC intra-enregistrement élevée ({wr['median']:.2f}) alors que l'AUC globale est faussée ({_auc(ys, ps):.2f})",
          wr["median"] > 0.95 and wr["n_rec"] == 4 and _auc(ys, ps) < wr["median"])
    check("within_recording_auc : aucun enregistrement avec assez de fenêtres -> None", within_recording_auc([1, 0], [0.5, 0.4], ["a", "a"], ["p", "p"]) is None)
    sc = pd.DataFrame(dict(patient=np.repeat(["a", "b", "c"], 40), y=np.tile(np.r_[np.ones(20), np.zeros(20)], 3).astype(int),
                           p=np.clip(np.tile(np.r_[np.full(20, .7), np.full(20, .2)], 3) + rng.normal(0, .1, 120), 0, 1)))
    out = diagnose_scores(sc)
    check("diagnose_scores : AUC par patient élevée et tableau des seuils", "médiane" in out[0] and len(out) == 9)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--scores_csv", nargs="*", default=None, help="un ou plusieurs m1_val_scores.csv produits par l'étape 3")
    ap.add_argument("--skip_baselines", action="store_true", help="ne pas recalculer les repères simples (analyse des fichiers de scores seulement)")
    ap.add_argument("--limit_per_role", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    main(a)
