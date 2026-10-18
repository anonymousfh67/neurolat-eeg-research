#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 4c -- Comparaison de plusieurs M2 (et de la régression logistique) AVEC INCERTITUDE.

Lit des fichiers de scores de validation (m2_val_scores.csv des réseaux, lr_raw_val_scores.csv de l'étape 4b), agrège par événement
(probabilités moyennes), puis calcule pour chaque modèle l'exactitude équilibrée par événement avec un intervalle de confiance
BOOTSTRAP PAR PATIENT (on rééchantillonne les patients, pas les événements : les événements d'un patient ne sont pas indépendants),
et la différence PAIRÉE avec le modèle de référence (même rééchantillonnage pour tous les modèles).
Si des noms se terminent par _seedK, le résumé par graine (moyenne, min, max) est aussi donné.

Usage :
    python step4c_compare_m2.py --ref .../lr_raw_val_scores.csv  .../m2_*/m2_val_scores.csv
    python step4c_compare_m2.py --selftest
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_train_m2 import event_aggregate, m2_metrics


def model_name(path):
    b = os.path.basename(path)
    return os.path.basename(os.path.dirname(os.path.abspath(path))) if b == "m2_val_scores.csv" else re.sub(r"_val_scores\.csv$", "", b)


def load_events(path):
    s = pd.read_csv(path)
    ev = event_aggregate(s.recording.values, s.eid.values, s.patient.values, s.y.values, s[["pL", "pR", "pB"]].values)
    return ev.sort_values(["rec", "eid"]).reset_index(drop=True)


def patient_counts(ev, patients):
    ok = (ev.pred.values == ev.y.values).astype(int)
    df = pd.DataFrame(dict(pid=ev.pid.values, y=ev.y.values, ok=ok))
    n = df.groupby(["pid", "y"]).size().unstack(fill_value=0).reindex(index=patients, columns=[0, 1, 2], fill_value=0).values
    c = df.groupby(["pid", "y"]).ok.sum().unstack(fill_value=0).reindex(index=patients, columns=[0, 1, 2], fill_value=0).values
    return c.astype(float), n.astype(float)


def bacc_from(c, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(n > 0, c / n, np.nan)
    return np.nanmean(rec, axis=-1), rec


def compare(paths, ref_path=None, n_boot=2000, seed=0):
    names = [model_name(p) for p in paths]
    evs = [load_events(p) for p in paths]
    ref_ev = load_events(ref_path) if ref_path else None
    labelled = ([("REF:" + model_name(ref_path), ref_ev)] if ref_ev is not None else []) + list(zip(names, evs))
    keys0 = list(zip(labelled[0][1].rec, labelled[0][1].eid))
    for _, e in labelled[1:]:
        if list(zip(e.rec, e.eid)) != keys0:
            raise SystemExit("Les modèles ne portent pas sur les mêmes événements de validation.")
    # ensembles : moyenne des probabilités de toutes les graines d'une même configuration (noms ..._seedK)
    groups = {}
    for name, ev in zip(names, evs):
        mt = re.match(r"(.+)_seed(\d+)$", name)
        if mt:
            groups.setdefault(mt.group(1), []).append(ev)
    for g, lst in groups.items():
        if len(lst) > 1:
            e = lst[0].copy()
            for col in ("pL", "pR", "pB"):
                e[col] = np.mean([x[col].values for x in lst], axis=0)
            e["pred"] = e[["pL", "pR", "pB"]].values.argmax(1)
            labelled.append(("ENS:" + g + f" ({len(lst)} graines)", e))
    patients = sorted(labelled[0][1].pid.unique())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(patients), (n_boot, len(patients)))
    stats = {}
    for name, ev in labelled:
        c, n = patient_counts(ev, patients)
        b_all, _ = bacc_from(c.sum(0), n.sum(0))
        bb, rb = bacc_from(c[idx].sum(1), n[idx].sum(1))
        m = m2_metrics(ev.y.values, ev.pred.values, ev.pid.values)
        stats[name] = dict(acc=m["acc"], bacc=float(b_all), ci=np.nanpercentile(bb, [2.5, 97.5]), recall=m["recall"],
                           rec_ci=np.nanpercentile(rb, [2.5, 97.5], axis=0), bacc_pw=m["bacc_pw"], boot=bb, n_ev=len(ev))
    return stats, len(patients), len(keys0)


def format_table(stats, n_pat, n_ev, ref_key=None):
    ref = stats[ref_key] if ref_key else None
    maj = None
    L = [f"validation : {n_ev} événements, {n_pat} patients (intervalles = bootstrap par patient, 95 %)",
         "modèle | exactitude | équilibrée [IC] | équilibrée par patient | rappels G/D/B | Δ équilibrée vs référence [IC]"]
    for name, st in stats.items():
        d = ""
        if ref is not None and st is not ref:
            diff = st["boot"] - ref["boot"]
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            d = f" | {st['bacc'] - ref['bacc']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        r = st["recall"]
        L.append(f"{name} | {st['acc']:.3f} | {st['bacc']:.3f} [{st['ci'][0]:.3f}, {st['ci'][1]:.3f}] | {st['bacc_pw']:.3f} | "
                 f"{r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f}{d}")
    groups = {}
    for name, st in stats.items():
        mt = re.match(r"(.+)_seed(\d+)$", name)
        if mt:
            groups.setdefault(mt.group(1), []).append(st["bacc"])
    for g, v in groups.items():
        if len(v) > 1:
            L.append(f"par graine, {g} : exactitude équilibrée moyenne {np.mean(v):.3f} (min {np.min(v):.3f}, max {np.max(v):.3f}, {len(v)} graines)")
    ens = [k for k in stats if k.startswith("ENS:")]
    for i in range(len(ens)):
        for j in range(i + 1, len(ens)):
            diff = stats[ens[i]]["boot"] - stats[ens[j]]["boot"]
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            L.append(f"différence appariée {ens[i]}  -  {ens[j]} : {stats[ens[i]]['bacc'] - stats[ens[j]]['bacc']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return L


def main(args):
    stats, n_pat, n_ev = compare(args.scores, args.ref, args.n_boot)
    ref_key = [k for k in stats if k.startswith("REF:")]
    L = ["<<<COPY_FROM_HERE>>>"] + format_table(stats, n_pat, n_ev, ref_key[0] if ref_key else None) + ["<<<COPY_TO_HERE>>>"]
    print("\n".join(L))


def selftest():
    import tempfile, shutil
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    try:
        rows = []
        for p in range(30):
            for e in range(rng.integers(2, 8)):
                rows.append((f"p{p:02d}_s001_t000", e, f"p{p:02d}", int(rng.integers(0, 3))))
        base = pd.DataFrame(rows, columns=["recording", "eid", "patient", "y"])

        def write(name, acc, seed):
            r = np.random.default_rng(seed)
            P = np.full((len(base), 3), 0.1)
            pred = np.where(r.random(len(base)) < acc, base.y.values, r.integers(0, 3, len(base)))
            P[np.arange(len(base)), pred] = 0.8
            d = base.copy()
            d["start_sample"] = 0
            d[["pL", "pR", "pB"]] = P
            os.makedirs(os.path.join(tmp, name), exist_ok=True)
            path = os.path.join(tmp, name, "m2_val_scores.csv") if name.startswith("m2_") else os.path.join(tmp, name + "_val_scores.csv")
            d.to_csv(path, index=False)
            return path

        ref = write("lr_raw", 0.5, 1)
        a0, a1 = write("m2_good_seed0", 0.95, 2), write("m2_good_seed1", 0.93, 3)
        bad = write("m2_bad_seed0", 0.5, 4)
        b0, b1 = write("m2_bad2_seed0", 0.5, 5), write("m2_bad2_seed1", 0.5, 6)
        stats, n_pat, n_ev = compare([a0, a1, bad, b0, b1], ref_path=ref, n_boot=1000)
        check(f"{n_pat} patients, {n_ev} événements chargés", n_pat == 30 and n_ev == len(base))
        g = stats["m2_good_seed0"]
        check(f"IC de l'exactitude équilibrée contient la valeur ({g['ci'][0]:.2f} <= {g['bacc']:.2f} <= {g['ci'][1]:.2f})", g["ci"][0] <= g["bacc"] <= g["ci"][1])
        L = format_table(stats, n_pat, n_ev, "REF:lr_raw")
        diff_good = stats["m2_good_seed0"]["boot"] - stats["REF:lr_raw"]["boot"]
        diff_bad = stats["m2_bad_seed0"]["boot"] - stats["REF:lr_raw"]["boot"]
        check("différence PAIRÉE : bon modèle, IC entièrement > 0", np.nanpercentile(diff_good, 2.5) > 0)
        check("différence PAIRÉE : modèle équivalent à la référence, IC contient 0", np.nanpercentile(diff_bad, 2.5) < 0 < np.nanpercentile(diff_bad, 97.5))
        check("résumé par graine présent pour m2_good (2 graines)", any("m2_good" in l and "2 graines" in l for l in L))
        k_good = [k for k in stats if k.startswith("ENS:m2_good")][0]
        k_bad = [k for k in stats if k.startswith("ENS:m2_bad2")][0]
        d_ens = stats[k_good]["boot"] - stats[k_bad]["boot"]
        check("ensembles de graines : bon modèle - mauvais modèle, IC entièrement > 0", np.nanpercentile(d_ens, 2.5) > 0)
        check("ligne « différence appariée » entre ensembles dans le tableau", any("différence appariée" in l and "ENS:m2_good" in l for l in L))
        # événements différents -> erreur explicite
        other = pd.read_csv(a0)
        other.loc[0, "eid"] = 99
        other.to_csv(os.path.join(tmp, "other_val_scores.csv"), index=False)
        try:
            compare([os.path.join(tmp, "other_val_scores.csv")], ref_path=ref)
            check("événements différents détectés", False)
        except SystemExit:
            check("événements différents détectés (arrêt avec erreur)", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="*", help="fichiers de scores de validation (m2_val_scores.csv ou *_val_scores.csv)")
    ap.add_argument("--ref", default=None, help="modèle de référence pour les différences appariées (ex. lr_raw_val_scores.csv)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.scores:
        ap.error("donnez au moins un fichier de scores")
    main(a)
