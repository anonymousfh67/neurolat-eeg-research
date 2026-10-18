#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 6 -- Comparateur simple pour M2 : régression logistique (5 bandes x 18 dérivations, avec miroir), appliquée à la MÊME règle de détection figée.

Pourquoi : savoir si le réseau M2 apporte quelque chose par rapport à un modèle très simple. La détection (M1, seuil, durée minimale) est identique ;
seul change le modèle qui donne gauche / droite / bilatéral. Deux variantes, TOUTES DEUX rapportées (aucune sélection sur eval) :
  lr_raw  : variables calculées sur le signal brut (garde l'amplitude et les rapports entre canaux)
  lr_norm : variables calculées sur le signal normalisé par canal (perd les rapports entre canaux)
Entraînement : uniquement les patients du rôle train (comme les réseaux), régularisation C = 1 (non réglée).

Deux commandes :
  fit   : entraîne les deux modèles sur les fenêtres de train du cache, écrit lr_models.pkl.
  score : relit chaque enregistrement DÉJÀ scoré par l'étape 5a (même fenêtres), calcule les probabilités LR et AJOUTE p2_lr_raw / p2_lr_norm
          au fichier .npz (les autres tableaux ne sont pas modifiés ; vérification que les fenêtres sont identiques).
          --role eval exige --final_eval et que params.json existe.
Ensuite :  step5b_pipeline_eval.py final --m2key lr_raw   (puis lr_norm)  applique la même règle figée avec ce M2.
"""
import argparse, os, pickle, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step2b_cache_windows as s2b
from step2b_cache_windows import index_files
from step3b_baselines_m1 import window_features
from step4b_baselines_m2 import fit_lr_model, predict_lr_model
from step5a_score_recordings import score_recording

VARIANTS = (("lr_raw", False), ("lr_norm", True))


def fit(args):
    from step3_train_m1 import load_cache
    from step3b_baselines_m1 import features_all
    from step4_train_m2 import m2_indices
    D = load_cache(args.cache_dir, args.split_csv, roles=("train",), verbose=False)
    itr = m2_indices(D, "train")
    _, _, bp, bpn = features_all(D)
    models = {"lr_raw": fit_lr_model(bp, D["m2"], D["pid"], itr), "lr_norm": fit_lr_model(bpn, D["m2"], D["pid"], itr)}
    os.makedirs(os.path.dirname(os.path.abspath(args.models)), exist_ok=True)
    with open(args.models, "wb") as f:
        pickle.dump(models, f)
    print("<<<COPY_FROM_HERE>>>")
    print(f"régressions logistiques entraînées sur {len(itr)} fenêtres de train ({len(set(D['pid'][itr]))} patients) -> {args.models}")
    print("<<<COPY_TO_HERE>>>")


def make_predictors(models, valid):
    """valid : masque des dérivations valides de l'enregistrement (comme à l'entraînement)."""
    return {k: (lambda x, k=k, nz=nz: predict_lr_model(models[k], window_features(x, valid, normalize=nz)[2])) for k, nz in VARIANTS}


def score(args):
    if args.role == "eval":
        if not args.final_eval:
            raise SystemExit("ERREUR : la partition eval exige --final_eval.")
        if not os.path.exists(args.params):
            raise SystemExit("ERREUR : params.json absent : la règle doit être figée avant de toucher à eval.")
    if not os.path.exists(args.models):
        raise SystemExit("ERREUR : lr_models.pkl absent : lancer d'abord `fit`.")
    with open(args.models, "rb") as f:
        models = pickle.load(f)
    man = pd.read_csv(args.manifest)
    man = man[(man.role == args.role) & (man.status == "ok")].drop_duplicates("recording", keep="last")
    idx = index_files(args.root, sorted(man.split.unique()))
    todo, n_done = [], 0
    for r in man.itertuples():
        if "p2_lr_raw" in np.load(r.score_path).files:
            n_done += 1
        else:
            todo.append(r)
    if args.limit:
        todo = todo[:args.limit]
    print(f"rôle {args.role} : {len(man)} enregistrements scorés, déjà faits {n_done}, à faire {len(todo)}", flush=True)
    t_all, n_err = time.time(), 0
    for k, r in enumerate(todo):
        try:
            sig, _ = s2b.load_electrodes(idx[r.recording]["edf"])
            deriv, valid = s2b.build_derivations(sig)
            z = dict(np.load(r.score_path))
            starts, flat, out = score_recording(deriv, valid, make_predictors(models, valid))
            if not (np.array_equal(starts, z["starts"]) and np.array_equal(flat, z["flat"])):
                raise RuntimeError("fenêtres différentes de celles de l'étape 5a")
            for name, _ in VARIANTS:
                z["p2_" + name] = out[name].astype(np.float32)
            np.savez(r.score_path + ".tmp.npz", **z)
            os.replace(r.score_path + ".tmp.npz", r.score_path)
        except Exception as e:
            n_err += 1
            print(f"  ERREUR {r.recording} : {type(e).__name__}: {str(e)[:100]}", flush=True)
        if (k + 1) % 20 == 0 or args.limit:
            print(f"  {k + 1}/{len(todo)} | {r.recording} | écoulé {time.time() - t_all:.0f} s", flush=True)
    print("<<<COPY_FROM_HERE>>>")
    print(f"rôle {args.role} : scorés {len(todo) - n_err} | erreurs {n_err} | déjà faits {n_done}")
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test (numpy + scikit-learn, sans EDF ni torch)
def selftest():
    import tempfile, shutil, io, contextlib
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    # modèles entraînés sur des variables synthétiques : gauche = bandes des canaux 0-7 plus fortes, droite = 8-15
    n = 300
    Xb = rng.normal(0, 1, (n, 90))
    y = rng.integers(0, 3, n)
    X3 = Xb.reshape(n, 18, 5)
    X3[y == 0, 0:8, :] += 2.5
    X3[y == 1, 8:16, :] += 2.5
    X3[y == 2, 0:16, :] += 2.5
    Xb = X3.reshape(n, 90)
    pid = rng.integers(0, 10, n)
    mdl = fit_lr_model(Xb, y, pid, np.arange(n))
    P = predict_lr_model(mdl, Xb)
    check(f"LR : probabilités (n, 3) de somme 1, exactitude {np.mean(P.argmax(1) == y):.2f} > 0,8 sur un cas facile",
          P.shape == (n, 3) and np.allclose(P.sum(1), 1) and np.mean(P.argmax(1) == y) > 0.8)
    # signal synthétique : la variante brute voit l'asymétrie d'amplitude, pas la normalisée (voir step4b) ; ici on vérifie seulement le branchement
    valid = np.ones(18, bool)
    deriv = (rng.standard_normal((18, 250 * 40)) * 10).astype(np.float32)
    deriv[0:8] *= 3
    models = {"lr_raw": mdl, "lr_norm": mdl}
    st, fl, out = score_recording(deriv, valid, make_predictors(models, valid), chunk=7)
    check("prédicteurs LR dans score_recording : deux sorties (n, 3), une ligne par fenêtre",
          set(out) == {"lr_raw", "lr_norm"} and out["lr_raw"].shape == (len(st), 3) and out["lr_norm"].shape == (len(st), 3))
    # score() sur un faux enregistrement : ajoute p2_lr_*, ne modifie pas le reste, refuse eval sans --final_eval / params.json
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, "edf", "eval", "pc000001", "s001", "01_tcp_ar")
        os.makedirs(d)
        for suf in (".edf", ".csv"):
            open(os.path.join(d, f"pc000001_s001_t000{suf}"), "w").write("x")
        sc_dir = os.path.join(tmp, "scores", "eval")
        os.makedirs(sc_dir)
        sp = os.path.join(sc_dir, "pc000001_s001_t000.npz")
        np.savez(sp, starts=st, p1=np.full(len(st), 0.3, np.float32), flat=fl, n_samples=deriv.shape[1], p2_wg=np.zeros((len(st), 3), np.float32))
        man = os.path.join(tmp, "scores", "scores_manifest.csv")
        pd.DataFrame([dict(role="eval", split="eval", recording="pc000001_s001_t000", patient="pc000001", status="ok", score_path=sp)]).to_csv(man, index=False)
        mp = os.path.join(tmp, "lr_models.pkl")
        pickle.dump(models, open(mp, "wb"))
        params = os.path.join(tmp, "params.json")
        s2b.load_electrodes = lambda p: ({e: deriv[0] for e in s2b.ELECTRODES}, 250.0)
        s2b.build_derivations = lambda sig: (deriv, valid)

        def A(**kw):
            a = dict(role="eval", root=os.path.join(tmp, "edf"), manifest=man, models=mp, params=params, final_eval=True, limit=None)
            a.update(kw)
            return argparse.Namespace(**a)

        def quiet(fn):
            with contextlib.redirect_stdout(io.StringIO()):
                return fn()

        for label, kw in (("sans --final_eval", dict(final_eval=False)), ("sans params.json", dict())):
            try:
                quiet(lambda: score(A(**kw)))
                check(f"eval {label} refusée", False)
            except SystemExit:
                check(f"eval {label} refusée", True)
        open(params, "w").write("{}")
        before = dict(np.load(sp))
        quiet(lambda: score(A()))
        after = np.load(sp)
        check("score : p2_lr_raw et p2_lr_norm ajoutés, (n, 3)", "p2_lr_raw" in after.files and "p2_lr_norm" in after.files and after["p2_lr_raw"].shape == (len(st), 3))
        check("score : p1, flat, starts, p2_wg inchangés", all(np.array_equal(before[k], after[k]) for k in before))
        t_before = os.path.getmtime(sp)
        quiet(lambda: score(A()))
        check("reprise : un enregistrement déjà fait n'est pas recalculé", os.path.getmtime(sp) == t_before)
        # fenêtres incohérentes -> erreur signalée, fichier intact
        z = dict(before)
        z["starts"] = st + 1
        np.savez(sp, **z)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            score(A())
        check("fenêtres différentes de la 5a : erreur signalée, rien d'écrit", "ERREUR" in buf.getvalue() and "p2_lr_raw" not in np.load(sp).files)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    print("NB : `fit` (cache) et la lecture réelle des EDF (mne) ne sont PAS couverts ici : lancer `score --role val --limit 3` dans Colab.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["fit", "score"])
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--role", choices=["val", "eval"])
    ap.add_argument("--root", default=os.environ.get("TUSZ_ROOT", "/content/drive/MyDrive/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf"))
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--manifest", default=base + "scores/scores_manifest.csv")
    ap.add_argument("--models", default=base + "scores/lr_models.pkl")
    ap.add_argument("--params", default=base + "scores/params.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--final_eval", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.cmd == "fit":
        fit(a)
    elif a.cmd == "score":
        if not a.role:
            ap.error("--role val|eval requis")
        score(a)
    else:
        ap.error("donner fit ou score")
