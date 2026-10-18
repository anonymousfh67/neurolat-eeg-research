#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 5a -- Scores M1 / M2 sur des ENREGISTREMENTS ENTIERS (fenêtres de 4 s tous les 2 s, du début à la fin).

  --role val  : enregistrements de VALIDATION (ceux du cache, patients du rôle val). Sert à choisir la règle de décision (5b tune).
  --role eval : enregistrements de la partition d'ÉVALUATION. Exige --final_eval, un params.json déjà figé sur la validation, et les MÊMES
                modèles que ceux enregistrés lors de l'exécution sur la validation (models.json).

Sorties : out_dir/<role>/<enregistrement>.npz (starts, p1, flat, n_samples, p2_<nom>) et out_dir/scores_manifest.csv (reprenable).
M1 : probabilité moyenne de crise sur les graines. M2 : probabilités moyennes (G/D/B) sur les graines, pour chaque ensemble nommé.

Usage :
  python step5a_score_recordings.py --role val --m1 D/m1_const_seed0 D/m1_const_seed1 D/m1_const_seed2 \
      --m2 windowglobal=D/m2_windowglobal_seed0,D/m2_windowglobal_seed1,D/m2_windowglobal_seed2 [--limit 3]
  python step5a_score_recordings.py --selftest
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step2b_cache_windows as s2b
from step2b_cache_windows import index_files, FS, WIN, STRIDE, MIN_VALID, flat_window_mask

MANIFEST_COLS = ["role", "split", "recording", "patient", "status", "reason", "n_windows", "flat_frac", "duration_s", "csv_path", "score_path", "seconds"]


def parse_m2(items):
    out = {}
    for it in items or []:
        name, dirs = it.split("=", 1)
        out[name] = [d for d in dirs.split(",") if d]
    return out


def list_recordings(role, cache_dir, split_csv, eval_step1_dir):
    if role == "val":
        man = pd.read_csv(os.path.join(cache_dir, "manifest.csv")).drop_duplicates("recording", keep="last")
        man = man[man.status == "ok"]
        sp = pd.read_csv(split_csv)[["patient", "role"]]
        r = man.merge(sp, on="patient")
        r = r[r.role == "val"][["split", "recording", "patient"]]
    else:
        rec = pd.read_csv(os.path.join(eval_step1_dir, "recordings.csv"))
        r = rec[rec.split == "eval"][["split", "recording", "patient"]]
    r = r.sort_values("recording").reset_index(drop=True)
    r.insert(0, "role", role)
    return r


def check_models_frozen(out_dir, role, m1, m2_sets):
    """Les modèles utilisés pour l'évaluation doivent être EXACTEMENT ceux de la validation (models.json)."""
    path = os.path.join(out_dir, "models.json")
    cur = dict(m1=list(m1), m2={n: list(d) for n, d in m2_sets.items()})
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old["m1"] != cur["m1"]:
            raise SystemExit("ERREUR : les modèles M1 ne sont pas ceux utilisés sur la validation (models.json).")
        for n, d in cur["m2"].items():
            if n in old["m2"] and old["m2"][n] != d:
                raise SystemExit(f"ERREUR : l'ensemble M2 '{n}' diffère de celui utilisé sur la validation (models.json).")
        cur["m2"] = {**old["m2"], **cur["m2"]}
    elif role == "eval":
        raise SystemExit("ERREUR : aucune exécution sur la validation (models.json absent) : on ne score pas eval avant.")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cur, f, indent=2)


def score_recording(deriv, valid, predictors, chunk=256):
    """Coeur numpy : fenêtres de 4 s tous les 2 s sur tout l'enregistrement ; retourne starts, masque plat, sorties de chaque prédicteur."""
    starts = np.arange(0, deriv.shape[1] - WIN + 1, STRIDE)
    out = {k: [] for k in predictors}
    flats = []
    for i in range(0, len(starts), chunk):
        xs = np.clip(np.stack([deriv[:, s:s + WIN] for s in starts[i:i + chunk]]), -60000, 60000).astype(np.float16)
        flats.append(flat_window_mask(xs, valid))
        for k, fn in predictors.items():
            out[k].append(np.asarray(fn(xs), np.float32))
    return starts, np.concatenate(flats), {k: np.concatenate(v) for k, v in out.items()}


def build_predictors(m1_dirs, m2_sets, device):
    """Charge les réseaux (torch) ; retourne {'m1': f, 'm2:<nom>': g} où f(x) -> (n,) et g(x) -> (n,3), moyennes sur les graines."""
    import torch
    from eeg_model import EEGConformer
    from step3_train_m1 import predict_proba_m1
    from step4_train_m2 import predict_proba_m2

    def load(d, n_classes, kind):
        with open(os.path.join(d, f"{kind}_config.json")) as f:
            cfg = json.load(f)
        m = EEGConformer(n_canaux=18, n_classes=n_classes, use_attn=not cfg.get("no_attn", False)).to(device)
        m.load_state_dict(torch.load(os.path.join(d, f"{kind}_best.pt"), map_location=device))
        m.eval()
        return m, cfg.get("norm", "window")

    m1s = [load(d, 2, "m1") for d in m1_dirs]
    preds = {"m1": lambda x: np.mean([predict_proba_m1(m, x, np.arange(len(x)), device, mode=mode) for m, mode in m1s], axis=0)}
    for name, dirs in m2_sets.items():
        ms = [load(d, 3, "m2") for d in dirs]
        preds["m2:" + name] = (lambda ms: (lambda x: np.mean([predict_proba_m2(m, x, np.arange(len(x)), device, mode=mode) for m, mode in ms], axis=0)))(ms)
        print(f"M2 '{name}' : {len(ms)} graines, normalisations {sorted({md for _, md in ms})}", flush=True)
    print(f"M1 : {len(m1s)} graines, normalisations {sorted({md for _, md in m1s})}", flush=True)
    return preds


def process_all(args, predictors):
    m2_sets = parse_m2(args.m2)
    if args.role == "eval":
        if not args.final_eval:
            raise SystemExit("ERREUR : la partition eval exige --final_eval (évaluation finale, une seule fois).")
        if not os.path.exists(args.params):
            raise SystemExit("ERREUR : params.json absent : choisir d'abord la règle de décision sur la validation (step5b tune).")
    check_models_frozen(args.out_dir, args.role, args.m1, m2_sets)
    recs = list_recordings(args.role, args.cache_dir, args.split_csv, args.eval_step1_dir)
    idx = index_files(args.root, sorted(recs.split.unique()))
    man_path = os.path.join(args.out_dir, "scores_manifest.csv")
    if os.path.exists(man_path):
        prev = pd.read_csv(man_path)
        done = set(prev[(prev.role == args.role) & prev.status.isin(["ok", "rejected"])].recording)
    else:
        done = set()
    todo = [r for r in recs.itertuples() if r.recording not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"rôle {args.role} : {len(recs)} enregistrements, déjà faits {len(done)}, à faire {len(todo)}", flush=True)
    if args.role == "eval":
        with open(os.path.join(args.out_dir, "eval_run_log.txt"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} eval lancée ; m1={args.m1} ; m2={m2_sets}\n")
    os.makedirs(os.path.join(args.out_dir, args.role), exist_ok=True)
    t_all = time.time()
    for k, r in enumerate(todo):
        t0 = time.time()
        row = dict(role=args.role, split=r.split, recording=r.recording, patient=r.patient, status="", reason="", n_windows=0, flat_frac=np.nan,
                   duration_s=np.nan, csv_path="", score_path="", seconds=0.0)
        try:
            f = idx.get(r.recording, {})
            row["csv_path"] = f.get("csv", "")
            if "edf" not in f or "csv" not in f:
                row.update(status="rejected", reason="fichier_edf_ou_csv_introuvable")
            else:
                sig, info = s2b.load_electrodes(f["edf"])
                if sig is None:
                    row.update(status="rejected", reason=info)
                else:
                    deriv, valid = s2b.build_derivations(sig)
                    row["duration_s"] = deriv.shape[1] / FS
                    if int(valid.sum()) < MIN_VALID:
                        row.update(status="rejected", reason=f"derivations_valides={int(valid.sum())}<{MIN_VALID}")
                    elif deriv.shape[1] < WIN:
                        row.update(status="rejected", reason="enregistrement_trop_court")
                    else:
                        starts, flat, out = score_recording(deriv, valid, predictors)
                        arrays = dict(starts=starts, p1=out["m1"], flat=flat, n_samples=deriv.shape[1])
                        arrays.update({"p2_" + k[3:]: v for k, v in out.items() if k.startswith("m2:")})
                        path = os.path.join(args.out_dir, args.role, r.recording + ".npz")
                        np.savez(path + ".tmp.npz", **arrays)
                        os.replace(path + ".tmp.npz", path)
                        row.update(status="ok", n_windows=len(starts), flat_frac=float(flat.mean()), score_path=path)
        except Exception as e:
            row.update(status="error", reason=f"{type(e).__name__}: {str(e)[:120]}")
        row["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame([row]).reindex(columns=MANIFEST_COLS).to_csv(man_path, mode="a", header=not os.path.exists(man_path), index=False)
        if (k + 1) % 10 == 0 or args.limit:
            print(f"  {k + 1}/{len(todo)} | {r.recording} | {row['status']} {row['reason']} | {row['seconds']} s | écoulé {time.time() - t_all:.0f} s", flush=True)
    m = pd.read_csv(man_path)
    m = m[m.role == args.role].drop_duplicates("recording", keep="last")
    print("<<<COPY_FROM_HERE>>>")
    print(f"rôle {args.role} : traités {len(m)} | ok {int((m.status == 'ok').sum())} | rejetés {int((m.status == 'rejected').sum())} | erreurs {int((m.status == 'error').sum())}")
    bad = m[m.status != "ok"]
    if len(bad):
        print("raisons :", bad.reason.fillna("").astype(str).str.replace(r"\d+", "N", regex=True).value_counts().to_dict())
    ok = m[m.status == "ok"]
    if len(ok):
        print(f"heures d'enregistrement scorées : {ok.duration_s.sum() / 3600:.1f} | fenêtres {int(ok.n_windows.sum())} | fraction plate moyenne {ok.flat_frac.mean():.4f} | durée médiane {ok.seconds.median():.1f} s/enregistrement")
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test (numpy, sans torch ni mne)
def selftest():
    import tempfile, shutil, io, contextlib
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    deriv = (rng.standard_normal((18, 250 * 60 + 123)) * 15).astype(np.float32)
    deriv[:, 250 * 20:250 * 30] = 0                                        # 10 s plates
    valid = np.ones(18, bool)
    preds = {"m1": lambda x: x.astype(np.float32).std(axis=(1, 2)) / 100, "m2:a": lambda x: np.tile([0.2, 0.3, 0.5], (len(x), 1))}
    st, fl, out = score_recording(deriv, valid, preds, chunk=256)
    st2, fl2, out2 = score_recording(deriv, valid, preds, chunk=7)
    check(f"fenêtres : début 0, pas de 2 s, dernière fenêtre complète ({len(st)})", st[0] == 0 and np.all(np.diff(st) == 500) and st[-1] + WIN <= deriv.shape[1] < st[-1] + WIN + 500)
    check("résultat indépendant de la taille des lots", np.array_equal(st, st2) and np.array_equal(fl, fl2) and np.allclose(out["m1"], out2["m1"]))
    check("fenêtres plates repérées (10 s nulles -> 4 fenêtres entièrement dans la zone nulle au moins)", fl.sum() >= 4 and fl[(st >= 250 * 21) & (st + WIN <= 250 * 29)].all())
    check("sorties de M2 : (n, 3)", out["m2:a"].shape == (len(st), 3))
    # parse_m2 et gel des modèles
    check("parse_m2", parse_m2(["wg=a,b,c", "k=d"]) == {"wg": ["a", "b", "c"], "k": ["d"]})
    tmp = tempfile.mkdtemp()
    try:
        # arbre de données synthétique + faux chargeur d'EDF
        for split, pats in (("train", ["pa000001"]), ("dev", ["pb000001"]), ("eval", ["pc000001", "pd000001"])):
            for p in pats:
                d = os.path.join(tmp, "edf", split, p, "s001", "01_tcp_ar")
                os.makedirs(d)
                for suf in (".edf", ".csv"):
                    open(os.path.join(d, f"{p}_s001_t000{suf}"), "w").write("x")
        cache = os.path.join(tmp, "cache")
        os.makedirs(cache)
        pd.DataFrame([dict(split="train", recording="pa000001_s001_t000", patient="pa000001", status="ok"),
                      dict(split="dev", recording="pb000001_s001_t000", patient="pb000001", status="ok")]).to_csv(os.path.join(cache, "manifest.csv"), index=False)
        pd.DataFrame([dict(patient="pa000001", role="train"), dict(patient="pb000001", role="val")]).to_csv(os.path.join(tmp, "split.csv"), index=False)
        e1 = os.path.join(tmp, "e1")
        os.makedirs(e1)
        pd.DataFrame([dict(split="eval", recording="pc000001_s001_t000", patient="pc000001"), dict(split="eval", recording="pd000001_s001_t000", patient="pd000001")]
                     ).to_csv(os.path.join(e1, "recordings.csv"), index=False)
        lv = list_recordings("val", cache, os.path.join(tmp, "split.csv"), e1)
        le = list_recordings("eval", cache, os.path.join(tmp, "split.csv"), e1)
        check("liste val : uniquement les patients du rôle val ; liste eval : uniquement eval, aucun recoupement",
              list(lv.patient) == ["pb000001"] and sorted(le.patient) == ["pc000001", "pd000001"] and not (set(lv.patient) & set(le.patient)))
        sig_ok = {e: (rng.standard_normal(250 * 30) * 15).astype(np.float32) for e in s2b.ELECTRODES}
        sig_bad = {e: v for e, v in sig_ok.items() if e not in ("T3", "T4", "T5")}
        calls = []
        s2b.load_electrodes = lambda p: (calls.append(p) or ((sig_bad if "pd000001" in p else sig_ok), 250.0))
        out_dir = os.path.join(tmp, "scores")

        def A(role, **kw):
            d = dict(role=role, root=os.path.join(tmp, "edf"), cache_dir=cache, split_csv=os.path.join(tmp, "split.csv"), eval_step1_dir=e1, out_dir=out_dir,
                     m1=["m1a", "m1b"], m2=["wg=m2a,m2b"], limit=None, final_eval=False, params=os.path.join(tmp, "params.json"))
            d.update(kw)
            return argparse.Namespace(**d)

        def quiet(fn):
            with contextlib.redirect_stdout(io.StringIO()):
                return fn()

        try:
            quiet(lambda: process_all(A("eval", final_eval=True), preds))
            check("eval refusée tant que la validation n'a pas été scorée (models.json absent)", False)
        except SystemExit:
            check("eval refusée tant que la validation n'a pas été scorée", True)
        quiet(lambda: process_all(A("val"), {"m1": preds["m1"], "m2:wg": preds["m2:a"]}))
        man = pd.read_csv(os.path.join(out_dir, "scores_manifest.csv"))
        check("val : 1 enregistrement ok, colonnes du manifeste", list(man.columns) == MANIFEST_COLS and list(man.status) == ["ok"])
        z = np.load(man.score_path.iloc[0])
        check("npz : starts, p1, flat, n_samples, p2_wg", sorted(z.files) == ["flat", "n_samples", "p1", "p2_wg", "starts"] and z["p2_wg"].shape[1] == 3)
        n_calls = len(calls)
        quiet(lambda: process_all(A("val"), {"m1": preds["m1"], "m2:wg": preds["m2:a"]}))
        check("reprise : une 2e exécution ne retraite rien", len(calls) == n_calls)
        try:
            quiet(lambda: process_all(A("eval"), preds))
            check("eval sans --final_eval refusée", False)
        except SystemExit:
            check("eval sans --final_eval refusée", True)
        try:
            quiet(lambda: process_all(A("eval", final_eval=True), preds))
            check("eval sans params.json refusée", False)
        except SystemExit:
            check("eval sans params.json refusée", True)
        open(os.path.join(tmp, "params.json"), "w").write("{}")
        try:
            quiet(lambda: process_all(A("eval", final_eval=True, m1=["autre"]), preds))
            check("eval avec d'autres modèles M1 refusée", False)
        except SystemExit:
            check("eval avec d'autres modèles M1 refusée (models.json)", True)
        quiet(lambda: process_all(A("eval", final_eval=True), {"m1": preds["m1"], "m2:wg": preds["m2:a"]}))
        man = pd.read_csv(os.path.join(out_dir, "scores_manifest.csv"))
        ev = man[man.role == "eval"].set_index("patient")
        check("eval : pc000001 ok, pd000001 rejeté (dérivations valides < 16), journal d'évaluation écrit",
              ev.loc["pc000001", "status"] == "ok" and ev.loc["pd000001", "status"] == "rejected" and "derivations_valides" in ev.loc["pd000001", "reason"]
              and os.path.exists(os.path.join(out_dir, "eval_run_log.txt")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    print("NB : le chargement des réseaux (torch) et la lecture des EDF (mne) ne sont PAS couverts ici : lancer d'abord --limit 3 dans Colab.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--role", choices=["val", "eval"])
    ap.add_argument("--root", default=os.environ.get("TUSZ_ROOT", "/content/drive/MyDrive/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf"))
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--eval_step1_dir", default=base + "etape1_eval")
    ap.add_argument("--out_dir", default=base + "scores")
    ap.add_argument("--params", default=base + "scores/params.json")
    ap.add_argument("--m1", nargs="+", help="dossiers des entraînements de M1 (une entrée par graine)")
    ap.add_argument("--m2", nargs="+", help="ensembles M2 : nom=dossier1,dossier2,...")
    ap.add_argument("--limit", type=int, default=None, help="ne traiter que N enregistrements (essai)")
    ap.add_argument("--final_eval", action="store_true", help="obligatoire pour --role eval")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not (a.role and a.m1 and a.m2):
        ap.error("--role, --m1 et --m2 sont obligatoires")
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", dev, flush=True)
    process_all(a, build_predictors(a.m1, parse_m2(a.m2), dev))
