#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 10 -- Autre famille de détecteurs : ARBRES DE GRADIENT sur variables calculées à la main + CONTEXTE de +-7 fenêtres. CPU seulement.

Motivation : sur TUSZ v2.0.3, un ensemble d'arbres sur des variables expertes avec contexte temporel atteint, selon son auteur, une sensibilité
de 0,75 pour 0,68 fausse alarme/h sur la partition eval (Scientific Reports, 2026 ; protocole de comptage différent du nôtre, donc indicatif).
Nos réseaux n'ont pas de contexte : chaque fenêtre de 4 s est jugée seule.

Pour chaque fenêtre de 4 s (pas de 2 s) : 24 variables de base (puissance de 5 bandes : moyenne et maximum sur les canaux ; asymétrie gauche/droite par bande ;
amplitude et line length : médiane et maximum ; fréquence dominante ; entropie spectrale ; Hjorth ; kurtosis). Contexte : pour chaque variable, moyenne et maximum
sur +-7 fenêtres, et écart au fond habituel de l'enregistrement (médiane des 5 minutes PRÉCÉDENTES) -> 96 variables.
Un seul modèle (HistGradientBoosting, paramètres fixés à l'avance, aucun réglage) entraîné sur TOUTES les fenêtres des enregistrements du rôle TRAIN.

Commandes (reprenables) :
  features --role train|val [--limit N]      calcule et enregistre les variables de base de chaque enregistrement (lit les EDF)
  features --role eval --final_eval          idem pour eval : exige le modèle ET une règle de décision déjà figée (params_v2.json de scores_gb)
  fit                                        entraîne sur train, affiche la qualité fenêtre par fenêtre sur la validation (toutes les fenêtres, distribution réelle)
  score --role val|eval [--final_eval]       écrit des scores au format de l'étape 5a (p1 = probabilité des arbres ; p2_* recopiés) dans scores_gb/
  ensemble --role val|eval [--final_eval]    moyenne des probabilités de M1 (scores/) et des arbres (scores_gb/) -> scores_ens/ (deux détecteurs d'information différente)
Puis :  step5b_pipeline_eval.py tune --manifest .../scores_gb/scores_manifest.csv --params .../scores_gb/params_v2.json [--smooths 1,3,5 --gaps 6,12,20]
"""
import argparse, os, pickle, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step2b_cache_windows as s2b
from step2b_cache_windows import index_files, seizure_intervals, m1_labels, FS, WIN, MIN_VALID
from step1_inventaire_labels import parse_annotation
from step5a_score_recordings import score_recording, list_recordings
from step8_hard_negatives import list_train_recordings

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 70)]
BASE_NAMES = ([f"bande{i}_moy" for i in range(5)] + [f"bande{i}_max" for i in range(5)] + [f"bande{i}_asym" for i in range(5)] +
              ["amp_med", "amp_max", "ll_med", "ll_max", "freq_dom", "entropie", "hjorth_mob", "hjorth_cplx", "kurtosis"])
N_BASE = len(BASE_NAMES)                  # 24
CTX_HALF = 7                              # +-7 fenêtres = +-14 s
PAST = 150                                # 150 fenêtres de 2 s = 5 minutes de fond
FEAT_COLS = ["role", "recording", "patient", "status", "reason", "n_windows", "seconds"]
GB_PARAMS = dict(max_depth=6, learning_rate=0.05, max_iter=300, l2_regularization=5.0, min_samples_leaf=200, random_state=0)
W_POS_MAX = 10.0


# ------------------------------------------------------------------ variables de base (numpy/scipy)
def base_features(x, valid):
    """x : (n, 18, T) en µV, valid : (18,) -> (n, 24) float32. Les dérivations invalides sont ignorées (nanmean / nanmax)."""
    from scipy.signal import welch
    from scipy.stats import kurtosis
    xf = x.astype(np.float32)[:, valid, :]
    n = len(xf)
    f, P = welch(xf, fs=FS, nperseg=250, axis=-1)                                      # (n, v, 126)
    bp_v = np.stack([np.log10(P[:, :, (f >= a) & (f < b)].mean(-1) + 1e-12) for a, b in BANDS], axis=2)       # (n, v, 5)
    bp = np.full((n, 18, len(BANDS)), np.nan, np.float32)
    bp[:, valid] = bp_v
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_b, max_b = np.nanmean(bp, axis=1), np.nanmax(bp, axis=1)
        asym = np.nanmean(np.abs(bp[:, :8] - bp[:, 8:16]), axis=1)                        # paires miroir : 0-7 <-> 8-15
    sd = xf.std(axis=-1)
    ll = np.abs(np.diff(xf, axis=-1)).mean(axis=-1)
    Pm = P.mean(axis=1)                                                                 # PSD moyenne sur les canaux
    band = (f >= 1) & (f <= 30)
    fd = f[band][Pm[:, band].argmax(axis=1)]
    q = Pm[:, band] / (Pm[:, band].sum(axis=1, keepdims=True) + 1e-12)
    ent = -(q * np.log(q + 1e-12)).sum(axis=1) / np.log(band.sum())
    d1 = np.diff(xf, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = xf.var(axis=-1) + 1e-12, d1.var(axis=-1) + 1e-12, d2.var(axis=-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    cplx = np.sqrt(v2 / v1) / mob
    out = np.concatenate([mean_b, max_b, asym, np.log10(np.median(sd, 1) + 1e-6)[:, None], np.log10(sd.max(1) + 1e-6)[:, None],
                          np.log10(np.median(ll, 1) + 1e-6)[:, None], np.log10(ll.max(1) + 1e-6)[:, None], fd[:, None], ent[:, None],
                          np.median(mob, 1)[:, None], np.median(cplx, 1)[:, None], np.median(kurtosis(xf, axis=-1), 1)[:, None]], axis=1)
    return out.astype(np.float32)


def context_matrix(B, flat):
    """B : (n, 24) variables de base d'un enregistrement (ordre du temps) -> (n, 96) : base, moyenne et maximum sur +-7 fenêtres,
    écart au fond habituel (médiane des 150 fenêtres PRÉCÉDENTES, la fenêtre courante exclue). Les fenêtres plates sont ignorées (NaN)."""
    Bn = np.where(flat[:, None], np.nan, B.astype(np.float64))
    df = pd.DataFrame(Bn)
    r = df.rolling(2 * CTX_HALF + 1, center=True, min_periods=1)
    past = df.rolling(PAST, min_periods=10).median().shift(1)
    return np.concatenate([Bn, r.mean().values, r.max().values, Bn - past.values], axis=1).astype(np.float32)


# ------------------------------------------------------------------ calcul des variables (lit les EDF ; mne requis)
def list_recs(role, args):
    if role == "train":
        return list_train_recordings(args.cache_dir, args.split_csv)
    return list_recordings(role, args.cache_dir, args.split_csv, args.eval_step1_dir)


def features_cmd(args):
    if args.role == "eval":
        if not args.final_eval:
            raise SystemExit("ERREUR : la partition eval exige --final_eval.")
        if not (os.path.exists(args.model) and os.path.exists(args.params)):
            raise SystemExit("ERREUR : modèle (gb_model.pkl) et règle de décision (params_v2.json de scores_gb) doivent exister AVANT de toucher à eval.")
    recs = list_recs(args.role, args)
    idx = index_files(args.root, sorted(recs.split.unique()))
    man_path = os.path.join(args.feat_dir, "manifest.csv")
    os.makedirs(args.feat_dir, exist_ok=True)
    done = set()
    if os.path.exists(man_path):
        prev = pd.read_csv(man_path).drop_duplicates(["role", "recording"], keep="last")
        done = set(prev[(prev.role == args.role) & prev.status.isin(["ok", "rejected"])].recording)
    todo = [r for r in recs.itertuples() if r.recording not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"rôle {args.role} : {len(recs)} enregistrements, déjà faits {len(done)}, à faire {len(todo)}", flush=True)
    t_all = time.time()
    for k, r in enumerate(todo):
        t0 = time.time()
        row = dict(role=args.role, recording=r.recording, patient=r.patient, status="", reason="", n_windows=0, seconds=0.0)
        try:
            f = idx.get(r.recording, {})
            if "edf" not in f or "csv" not in f:
                row.update(status="rejected", reason="fichier_edf_ou_csv_introuvable")
            else:
                sig, info = s2b.load_electrodes(f["edf"])
                if sig is None:
                    row.update(status="rejected", reason=str(info))
                else:
                    deriv, valid = s2b.build_derivations(sig)
                    if int(valid.sum()) < MIN_VALID or deriv.shape[1] < WIN:
                        row.update(status="rejected", reason="derivations_insuffisantes_ou_trop_court")
                    else:
                        _, rows = parse_annotation(f["csv"])
                        starts, flat, out = score_recording(deriv, valid, {"f": lambda x: base_features(x, valid)})
                        m1 = m1_labels(starts / FS, seizure_intervals(rows))
                        path = os.path.join(args.feat_dir, args.role, r.recording + ".npz")
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        np.savez(path + ".tmp.npz", B=out["f"], starts=starts, flat=flat, m1=m1, n_samples=deriv.shape[1])
                        os.replace(path + ".tmp.npz", path)
                        row.update(status="ok", n_windows=len(starts))
        except Exception as e:
            row.update(status="error", reason=f"{type(e).__name__}: {str(e)[:120]}")
        row["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame([row]).reindex(columns=FEAT_COLS).to_csv(man_path, mode="a", header=not os.path.exists(man_path), index=False)
        if (k + 1) % 20 == 0 or args.limit:
            print(f"  {k + 1}/{len(todo)} | {r.recording} | {row['status']} {row['reason']} | écoulé {time.time() - t_all:.0f} s", flush=True)
    m = pd.read_csv(man_path).drop_duplicates(["role", "recording"], keep="last")
    m = m[m.role == args.role]
    print("<<<COPY_FROM_HERE>>>")
    print(f"variables, rôle {args.role} : ok {int((m.status == 'ok').sum())} | rejetés {int((m.status == 'rejected').sum())} | erreurs {int((m.status == 'error').sum())} | "
          f"fenêtres {int(m[m.status == 'ok'].n_windows.sum())}")
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ entraînement
def load_role(feat_dir, role):
    """Matrice (n, 96), labels M1 (0/1/-1), patient (str) et enregistrement de toutes les fenêtres non plates d'un rôle."""
    man = pd.read_csv(os.path.join(feat_dir, "manifest.csv")).drop_duplicates(["role", "recording"], keep="last")
    man = man[(man.role == role) & (man.status == "ok")].sort_values("recording")
    X, y, pat, rec = [], [], [], []
    for r in man.itertuples():
        z = np.load(os.path.join(feat_dir, role, r.recording + ".npz"))
        keep = ~z["flat"]
        Xr = context_matrix(z["B"], z["flat"])
        X.append(Xr[keep]); y.append(z["m1"][keep]); pat += [r.patient] * int(keep.sum()); rec += [r.recording] * int(keep.sum())
    return np.concatenate(X), np.concatenate(y), np.array(pat), np.array(rec)


def fit_model(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    k = y >= 0                                                       # les fenêtres ambiguës (chevauchement partiel d'une crise) ne servent pas à l'entraînement
    X, y = X[k], y[k].astype(int)
    w = np.where(y == 1, min(W_POS_MAX, (y == 0).sum() / max(1, (y == 1).sum())), 1.0)
    return HistGradientBoostingClassifier(**GB_PARAMS).fit(X, y, sample_weight=w)


def fit_cmd(args):
    from step3_train_m1 import val_metrics
    X, y, pat, _ = load_role(args.feat_dir, "train")
    print(f"train : {len(y)} fenêtres, {len(set(pat))} patients, crise {int((y == 1).sum())} ({(y == 1).mean():.3f}), ambiguës écartées {int((y < 0).sum())}", flush=True)
    t0 = time.time()
    model = fit_model(X, y)
    os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)
    pickle.dump(model, open(args.model, "wb"))
    L = ["<<<COPY_FROM_HERE>>>", f"modèle entraîné en {time.time() - t0:.0f} s sur {int((y >= 0).sum())} fenêtres ({len(set(pat))} patients) -> {args.model}"]
    if os.path.exists(os.path.join(args.feat_dir, "manifest.csv")) and (pd.read_csv(os.path.join(args.feat_dir, "manifest.csv")).role == "val").any():
        Xv, yv, pv, _ = load_role(args.feat_dir, "val")
        k = yv >= 0
        codes = pd.factorize(pv[k])[0]
        p = model.predict_proba(Xv[k])[:, 1]
        m = val_metrics(yv[k], p, codes)
        L.append(f"validation, TOUTES les fenêtres non plates (distribution réelle, {int(k.sum())} fenêtres, {len(set(pv))} patients, crise {(yv[k] == 1).mean():.3f}) : "
                 f"AUC {m['auc']:.4f} | AUC par patient {m['auc_pw']:.4f}")
        L.append("(à comparer à M1 sur les mêmes conditions : voir l'étape 5b tune ; l'AUC de M1 de 0,85 portait sur un jeu de validation à 79 % de crises, non comparable)")
    L.append("<<<COPY_TO_HERE>>>")
    print("\n".join(L))


# ------------------------------------------------------------------ scores au format de l'étape 5a
def score_cmd(args):
    if args.role == "eval":
        if not args.final_eval:
            raise SystemExit("ERREUR : la partition eval exige --final_eval.")
        if not os.path.exists(args.params):
            raise SystemExit("ERREUR : règle de décision figée (scores_gb/params_v2.json) absente : lancer `tune` sur la validation d'abord.")
    if not os.path.exists(args.model):
        raise SystemExit("ERREUR : gb_model.pkl absent : lancer `fit` d'abord.")
    model = pickle.load(open(args.model, "rb"))
    src = pd.read_csv(args.src_manifest)
    src = src[src.role == args.role].drop_duplicates("recording", keep="last")
    feats = pd.read_csv(os.path.join(args.feat_dir, "manifest.csv")).drop_duplicates(["role", "recording"], keep="last")
    feats = set(feats[(feats.role == args.role) & (feats.status == "ok")].recording)
    os.makedirs(os.path.join(args.out_dir, args.role), exist_ok=True)
    rows, n_ok, n_missing = [], 0, 0
    for r in src.itertuples():
        row = r._asdict()
        row.pop("Index", None)
        if r.status == "ok":
            if r.recording not in feats:
                row.update(status="error", reason="variables_absentes", score_path="")
                n_missing += 1
            else:
                z0 = np.load(r.score_path)
                zf = np.load(os.path.join(args.feat_dir, args.role, r.recording + ".npz"))
                if not np.array_equal(z0["starts"], zf["starts"]) or not np.array_equal(z0["flat"], zf["flat"]):
                    raise SystemExit(f"ERREUR : fenêtres différentes entre les scores 5a et les variables pour {r.recording}")
                p = model.predict_proba(context_matrix(zf["B"], zf["flat"]))[:, 1].astype(np.float32)
                arrays = {k: z0[k] for k in z0.files if k != "p1"}
                arrays["p1"] = p
                path = os.path.join(args.out_dir, args.role, r.recording + ".npz")
                np.savez(path + ".tmp.npz", **arrays)
                os.replace(path + ".tmp.npz", path)
                row["score_path"] = path
                n_ok += 1
        rows.append(row)
    new = pd.DataFrame(rows)
    man_path = os.path.join(args.out_dir, "scores_manifest.csv")
    if os.path.exists(man_path):                                          # on conserve les lignes de l'autre rôle
        old = pd.read_csv(man_path)
        new = pd.concat([old[old.role != args.role], new], ignore_index=True)
    new.to_csv(man_path, index=False)
    print("<<<COPY_FROM_HERE>>>")
    print(f"scores des arbres, rôle {args.role} : {n_ok} enregistrements écrits dans {args.out_dir} | variables manquantes {n_missing}")
    print("<<<COPY_TO_HERE>>>")


def ensemble_cmd(args):
    """p1 = moyenne de p1(M1) et p1(arbres), mêmes fenêtres ; p2_* de M1. Aucune nouvelle lecture d'EDF."""
    if args.role == "eval":
        if not args.final_eval:
            raise SystemExit("ERREUR : la partition eval exige --final_eval.")
        if not os.path.exists(args.params_ens):
            raise SystemExit("ERREUR : règle de décision figée de l'ensemble (scores_ens/params_v2.json) absente : lancer `tune` sur la validation d'abord.")
    a = pd.read_csv(args.src_manifest)
    a = a[a.role == args.role].drop_duplicates("recording", keep="last")
    b = pd.read_csv(os.path.join(args.out_dir_gb, "scores_manifest.csv"))
    b = b[b.role == args.role].drop_duplicates("recording", keep="last").set_index("recording")
    os.makedirs(os.path.join(args.out_dir, args.role), exist_ok=True)
    rows, n_ok = [], 0
    for r in a.itertuples():
        row = r._asdict()
        row.pop("Index", None)
        if r.status == "ok":
            if r.recording not in b.index or b.loc[r.recording, "status"] != "ok":
                raise SystemExit(f"ERREUR : {r.recording} absent des scores des arbres : lancer `score` d'abord pour ce rôle.")
            z0, z1 = np.load(r.score_path), np.load(b.loc[r.recording, "score_path"])
            if not np.array_equal(z0["starts"], z1["starts"]) or not np.array_equal(z0["flat"], z1["flat"]):
                raise SystemExit(f"ERREUR : fenêtres différentes pour {r.recording}")
            arrays = {k: z0[k] for k in z0.files}
            arrays["p1"] = ((z0["p1"] + z1["p1"]) / 2).astype(np.float32)
            path = os.path.join(args.out_dir, args.role, r.recording + ".npz")
            np.savez(path + ".tmp.npz", **arrays)
            os.replace(path + ".tmp.npz", path)
            row["score_path"] = path
            n_ok += 1
        rows.append(row)
    new = pd.DataFrame(rows)
    man_path = os.path.join(args.out_dir, "scores_manifest.csv")
    if os.path.exists(man_path):
        old = pd.read_csv(man_path)
        new = pd.concat([old[old.role != args.role], new], ignore_index=True)
    new.to_csv(man_path, index=False)
    print("<<<COPY_FROM_HERE>>>")
    print(f"ensemble M1 + arbres, rôle {args.role} : {n_ok} enregistrements écrits dans {args.out_dir}")
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test (numpy, scipy, scikit-learn ; sans EDF)
def selftest():
    import tempfile, shutil, io, contextlib
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    valid = np.ones(18, bool)
    # 1) variables de base : forme, valeurs finies, sensibilité à l'amplitude et à l'asymétrie
    x = (rng.standard_normal((6, 18, 1000)) * 10).astype(np.float32)
    B = base_features(x, valid)
    check(f"base_features : forme (6, {N_BASE}) et valeurs finies", B.shape == (6, N_BASE) and np.isfinite(B).all())
    B2 = base_features(x * 4, valid)
    check("amplitude x4 : log10 de l'amplitude médiane augmente de ~0,6", abs((B2[:, BASE_NAMES.index('amp_med')] - B[:, BASE_NAMES.index('amp_med')]).mean() - np.log10(4)) < 0.05)
    xa = x.copy(); xa[:, :8] *= 4                                            # hémisphère gauche plus ample
    Ba = base_features(xa, valid)
    i_as = [BASE_NAMES.index(f"bande{i}_asym") for i in range(5)]
    check("asymétrie G/D : variables d'asymétrie plus grandes quand un hémisphère est x4", (Ba[:, i_as].mean() > 3 * B[:, i_as].mean()))
    v2 = valid.copy(); v2[3] = False
    check("dérivation invalide ignorée sans produire de NaN", np.isfinite(base_features(x, v2)).all())
    t = np.arange(1000) / FS
    xs = np.tile(np.sin(2 * np.pi * 10 * t)[None, None, :], (3, 18, 1)).astype(np.float32) * 20 + x[:3] * 0.01
    check("fréquence dominante d'un sinus de 10 Hz = 10 Hz", np.allclose(base_features(xs, valid)[:, BASE_NAMES.index("freq_dom")], 10.0, atol=1.0))
    # 2) contexte : forme, causalité du fond, centrage de la fenêtre glissante
    n = 400
    Bm = rng.normal(0, 1, (n, N_BASE)).astype(np.float32)
    flat = np.zeros(n, bool)
    C = context_matrix(Bm, flat)
    check("context_matrix : (n, 96)", C.shape == (n, 4 * N_BASE))
    Bm2 = Bm.copy(); Bm2[300:] += 5.0                                          # on change l'avenir
    C2 = context_matrix(Bm2, flat)
    j = 3 * N_BASE                                                             # bloc « écart au fond »
    check("écart au fond (5 min précédentes) : ne dépend pas de l'avenir", np.allclose(C[:290, j:], C2[:290, j:], equal_nan=True))
    check("moyenne sur +-7 fenêtres : une valeur isolée influence exactement 15 fenêtres", ((context_matrix(np.where(np.arange(n)[:, None] == 200, 100.0, 0.0) * np.ones((n, N_BASE)), flat)[:, N_BASE] > 1).sum() == 15))
    fl = flat.copy(); fl[100:104] = True
    Cf = context_matrix(Bm, fl)
    check("fenêtres plates : variables de base NaN, contexte calculé sur les voisines", np.isnan(Cf[100, :N_BASE]).all() and np.isfinite(Cf[110, N_BASE:2 * N_BASE]).all())
    # 3) de bout en bout sur des enregistrements synthétiques : fit, score, protections
    tmp = tempfile.mkdtemp()
    try:
        fd = os.path.join(tmp, "feat"); od = os.path.join(tmp, "scores_gb"); sd = os.path.join(tmp, "scores")
        for d in (os.path.join(fd, "train"), os.path.join(fd, "val"), os.path.join(sd, "val"), os.path.join(sd, "eval")):
            os.makedirs(d)
        fm, sm = [], []
        for role, pats in (("train", [f"t{i:02d}" for i in range(12)]), ("val", [f"v{i:02d}" for i in range(4)])):
            for pa in pats:
                for rk in range(2):
                    name = f"{pa}_s001_t00{rk}"
                    nw = 400
                    Bz = rng.normal(0, 1, (nw, N_BASE)).astype(np.float32)
                    m1 = np.zeros(nw, np.int8); m1[150:200] = 1
                    Bz[150:200, BASE_NAMES.index("amp_med")] += 3.0                       # une crise = amplitude plus forte, durable
                    starts = np.arange(nw) * 500
                    np.savez(os.path.join(fd, role, name + ".npz"), B=Bz, starts=starts, flat=np.zeros(nw, bool), m1=m1, n_samples=nw * 500 + 500)
                    fm.append(dict(role=role, recording=name, patient=pa, status="ok", reason="", n_windows=nw, seconds=1.0))
                    if role == "val":
                        sp = os.path.join(sd, "val", name + ".npz")
                        np.savez(sp, starts=starts, p1=np.zeros(nw, np.float32), flat=np.zeros(nw, bool), n_samples=nw * 500 + 500, p2_wg=np.ones((nw, 3), np.float32) / 3)
                        sm.append(dict(role="val", split="dev", recording=name, patient=pa, status="ok", reason="", csv_path="c.csv", score_path=sp))
        sm.append(dict(role="val", split="dev", recording="v00_s001_t009", patient="v00", status="rejected", reason="x", csv_path="c.csv", score_path=""))
        pd.DataFrame(fm).to_csv(os.path.join(fd, "manifest.csv"), index=False)
        pd.DataFrame(sm).to_csv(os.path.join(sd, "scores_manifest.csv"), index=False)
        A = argparse.Namespace(feat_dir=fd, model=os.path.join(tmp, "gb.pkl"), params=os.path.join(od, "params_v2.json"), out_dir=od,
                               src_manifest=os.path.join(sd, "scores_manifest.csv"), role="val", final_eval=False)

        def quiet(fn):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fn()
            return buf.getvalue()
        out = quiet(lambda: fit_cmd(A))
        auc = float(out.split("AUC par patient")[1].split()[0]) if "AUC par patient" in out else 0.0
        check(f"fit : AUC par patient sur la validation synthétique {auc:.3f} > 0,95", auc > 0.95)
        quiet(lambda: score_cmd(A))
        z = np.load(os.path.join(od, "val", "v00_s001_t000.npz"))
        z0 = np.load(os.path.join(sd, "val", "v00_s001_t000.npz"))
        check("score : p1 remplacé par les arbres (fort pendant la crise), starts/flat/p2 recopiés, rejeté conservé dans le manifeste",
              z["p1"][160:190].mean() > 0.7 and z["p1"][:100].mean() < 0.2 and np.array_equal(z["starts"], z0["starts"]) and np.array_equal(z["p2_wg"], z0["p2_wg"])
              and (pd.read_csv(os.path.join(od, "scores_manifest.csv")).status == "rejected").sum() == 1)
        try:
            quiet(lambda: score_cmd(argparse.Namespace(**dict(vars(A), role="eval"))))
            check("score eval sans --final_eval refusé", False)
        except SystemExit:
            check("score eval sans --final_eval refusé", True)
        try:
            quiet(lambda: score_cmd(argparse.Namespace(**dict(vars(A), role="eval", final_eval=True))))
            check("score eval sans règle figée refusé", False)
        except SystemExit:
            check("score eval sans règle figée refusé", True)
        E = argparse.Namespace(role="val", final_eval=False, src_manifest=os.path.join(sd, "scores_manifest.csv"), out_dir_gb=od,
                               out_dir=os.path.join(tmp, "scores_ens"), params_ens=os.path.join(tmp, "scores_ens", "params_v2.json"))
        quiet(lambda: ensemble_cmd(E))
        ze = np.load(os.path.join(E.out_dir, "val", "v00_s001_t000.npz"))
        check("ensemble : p1 = moyenne de M1 (ici 0) et des arbres, p2 recopiés",
              np.allclose(ze["p1"], z["p1"] / 2, atol=1e-6) and np.array_equal(ze["p2_wg"], z0["p2_wg"]))
        try:
            quiet(lambda: ensemble_cmd(argparse.Namespace(**dict(vars(E), role="eval", final_eval=True))))
            check("ensemble eval sans règle figée refusé", False)
        except SystemExit:
            check("ensemble eval sans règle figée refusé", True)
        B_args = argparse.Namespace(role="eval", final_eval=True, model=os.path.join(tmp, "absent.pkl"), params=os.path.join(tmp, "absent.json"))
        try:
            features_cmd(B_args)
            check("variables eval sans modèle ni règle figés refusées", False)
        except SystemExit:
            check("variables eval sans modèle ni règle figés refusées", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    print("NB : la lecture réelle des EDF (mne) n'est PAS couverte ici : lancer d'abord `features --role val --limit 3` dans Colab.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["features", "fit", "score", "ensemble"])
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--role", choices=["train", "val", "eval"])
    ap.add_argument("--root", default=os.environ.get("TUSZ_ROOT", "/content/drive/MyDrive/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf"))
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--eval_step1_dir", default=base + "etape1_eval")
    ap.add_argument("--feat_dir", default=base + "gb_features")
    ap.add_argument("--model", default=base + "gb/gb_model.pkl")
    ap.add_argument("--src_manifest", default=base + "scores/scores_manifest.csv", help="scores de l'étape 5a dont on recopie starts / flat / p2_*")
    ap.add_argument("--out_dir", default=base + "scores_gb")
    ap.add_argument("--params", default=base + "scores_gb/params_v2.json", help="règle de décision figée (écrite par step5b tune sur scores_gb)")
    ap.add_argument("--out_dir_ens", default=base + "scores_ens")
    ap.add_argument("--params_ens", default=base + "scores_ens/params_v2.json", help="règle figée de l'ensemble (écrite par step5b tune sur scores_ens)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--final_eval", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.cmd == "features":
        if a.role is None:
            ap.error("--role requis")
        features_cmd(a)
    elif a.cmd == "fit":
        fit_cmd(a)
    elif a.cmd == "score":
        if a.role not in ("val", "eval"):
            ap.error("--role val|eval requis")
        score_cmd(a)
    elif a.cmd == "ensemble":
        if a.role not in ("val", "eval"):
            ap.error("--role val|eval requis")
        a.out_dir_gb, a.out_dir = a.out_dir, a.out_dir_ens
        ensemble_cmd(a)
    else:
        ap.error("donner features, fit ou score")
