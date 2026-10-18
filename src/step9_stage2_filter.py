#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 9 -- Second étage de décision (idée d'ALARM-Net, préprint 2026) : filtrer les fausses alarmes au niveau de l'ÉVÉNEMENT. CPU seulement.

Constat : M1 donne une probabilité par fenêtre ; la règle actuelle la compare à un seuil. Mais une vraie crise a une FORME (probabilité élevée et
soutenue, qui monte) et une fausse alarme aussi (pic bref, artéfact isolé, enregistrement agité). Un petit modèle d'arbres, très régularisé, apprend à
distinguer les deux à partir de la forme de la probabilité de M1 autour de chaque détection.

  Étage 1 : « propositions » = détection avec un seuil bas tau_p et une fusion des trous <= gap (très sensible).
  Étage 2 : modèle d'arbres sur 10 variables par proposition (maximum et moyenne de p, durée, densité, début, contexte de 60 s et de 5 min AVANT,
            historique des alarmes précédentes : toutes causales, aucune ne regarde l'avenir) ; la proposition est gardée si score >= tau2.
  Un modèle de M2 (gauche/droite/bilatéral) étiquette ensuite chaque détection gardée, comme avant.

PROTOCOLE (aucune fuite vers eval) :
  * tune  : le second étage est entraîné UNIQUEMENT sur les enregistrements de VALIDATION, avec validation croisée PAR PATIENT (5 plis) : le score de chaque
            proposition vient d'un modèle qui n'a jamais vu son patient. La configuration (tau_p, gap, tau2) est choisie sur ces scores hors pli, puis
            comparée à la règle à un étage, avec un intervalle de confiance apparié (bootstrap par patient). Écrit params_stage2.json et stage2.pkl
            (modèle réentraîné sur toute la validation).
  * final : applique la configuration FIGÉE et le modèle figé à la partition eval (exige --final_eval). C'est une SECONDE évaluation sur eval :
            elle doit être déclarée comme telle dans le papier (la première est celle de step5b final).
Grille déclarée à l'avance : tau_p dans {0.2, 0.3, 0.4, 0.5} x gap dans {6, 20, 40} s x tau2 dans {0.05, ..., 0.95} = 12 configurations.

Usage :  python step9_stage2_filter.py tune      |      python step9_stage2_filter.py final --final_eval      |      --selftest
"""
import argparse, json, os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2b_cache_windows import FS, WIN
from step5b_pipeline_eval import (detect, eval_dets, load_split, metrics_from_sums, fmt_final, fmt_oracle, per_patient_oracle, per_patient_sums,
                                  bootstrap, tune as tune_stage1, PRE_TOL_S, MERGE_GAP_S)

TAU_P = (0.2, 0.3, 0.4, 0.5)
GAPS = (6.0, 20.0, 40.0)
TAU2 = np.round(np.arange(0.05, 0.96, 0.05), 2)
N_FOLDS = 5
FEATURE_NAMES = ["p_max", "p_moy", "n_fenetres", "duree_s", "densite", "frac_p>=0.8", "p_debut(3)", "ctx60_max", "ctx60_moy", "ctx300_moy", "n_prop_10min", "ecart_prec_s"]
HIT = lambda d, a, b: a - PRE_TOL_S <= d["start"] <= b


def proposals(sc, tau_p, gap):
    return detect(sc["starts"], sc["p1"], sc["flat"], tau_p, 1, gap)


def features(sc, dets):
    """Une ligne de variables par proposition. Toutes causales : le contexte vient uniquement de ce qui précède le début de la proposition."""
    t0 = sc["starts"] / FS
    t1 = t0 + WIN / FS
    p1, ok = sc["p1"], ~sc["flat"]
    rows = []
    prev_end, prev_starts = -1e9, []
    for d in dets:
        span = (t0 >= d["start"] - 1e-9) & (t1 <= d["stop"] + 1e-9) & ok
        ps = p1[span] if span.any() else p1[d["idx"]]
        pre60 = (t1 <= d["start"] + 1e-9) & (t0 >= d["start"] - 60) & ok
        pre300 = (t1 <= d["start"] + 1e-9) & (t0 >= d["start"] - 300) & ok
        rows.append([ps.max(), ps.mean(), len(d["idx"]), d["stop"] - d["start"], len(d["idx"]) / max(1, int(span.sum())), float((ps >= 0.8).mean()),
                     ps[:3].mean(), p1[pre60].max() if pre60.any() else 0.0, p1[pre60].mean() if pre60.any() else 0.0,
                     p1[pre300].mean() if pre300.any() else 0.0, sum(1 for s in prev_starts if d["start"] - 600 <= s < d["start"]),
                     min(d["start"] - prev_end, 3600.0)])
        prev_end = d["stop"]
        prev_starts.append(d["start"])
    return np.array(rows, float).reshape(len(dets), len(FEATURE_NAMES))


def build(recs, tau_p, gap):
    """Pour chaque enregistrement : propositions, variables, et vrai/faux de chaque proposition (début dans [début de crise - 30 s, fin])."""
    out = []
    for pat, sc, S, E in recs:
        dets = proposals(sc, tau_p, gap)
        y = np.array([any(HIT(d, a, b) for a, b in S) for d in dets], int)
        out.append(dict(pat=pat, sc=sc, S=S, E=E, dets=dets, X=features(sc, dets), y=y))
    return out


def new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100, l2_regularization=10.0, min_samples_leaf=30, random_state=0)


def fit_stage2(items):
    X = np.concatenate([i["X"] for i in items])
    y = np.concatenate([i["y"] for i in items])
    if len(y) == 0 or y.min() == y.max():
        return None                                          # une seule classe : pas de modèle (toutes les propositions sont gardées)
    return new_model().fit(X, y)


def score_items(model, items):
    for i in items:
        i["s"] = np.ones(len(i["dets"])) if (model is None or len(i["dets"]) == 0) else model.predict_proba(i["X"])[:, 1]


def patient_folds(patients, n_folds=N_FOLDS, seed=0):
    pats = np.array(sorted(set(patients)))
    perm = np.random.default_rng(seed).permutation(len(pats))
    return {pats[j]: k % n_folds for k, j in enumerate(perm)}


def oof_scores(items, n_folds=N_FOLDS):
    """Score hors pli : chaque proposition est notée par un modèle entraîné sur d'autres patients."""
    fold = patient_folds([i["pat"] for i in items], n_folds)
    for k in range(n_folds):
        tr = [i for i in items if fold[i["pat"]] != k]
        te = [i for i in items if fold[i["pat"]] == k]
        score_items(fit_stage2(tr), te)


def sums_at(items, tau2, m2key):
    acc = {}
    for i in items:
        dets = [d for d, s in zip(i["dets"], i["s"]) if s >= tau2]
        acc[i["pat"]] = acc.get(i["pat"], 0) + eval_dets(i["sc"], i["S"], i["E"], dets, m2key)
    pats = sorted(acc)
    return pats, np.array([acc[p] for p in pats])


def paired_diff(Ma, Mb, n_boot=2000, seed=0):
    """Différences (a - b) de F1, sensibilité et fausses alarmes par heure, bootstrap PAR PATIENT (mêmes patients rééchantillonnés pour a et b)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(Ma), (n_boot, len(Ma)))
    A, B = metrics_from_sums(Ma[idx].sum(1)), metrics_from_sums(Mb[idx].sum(1))
    pa, pb = metrics_from_sums(Ma.sum(0)), metrics_from_sums(Mb.sum(0))
    L = []
    for k in ("F1", "sensibilité", "fausses_alarmes_par_h", "précision"):
        d = A[k] - B[k]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        L.append(f"  {k:24s} {float(pa[k] - pb[k]):+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    return L


def tune(recs, m2key, taus2=TAU2):
    rows, best, best_items = [], None, None
    for tau_p in TAU_P:
        for gap in GAPS:
            items = build(recs, tau_p, gap)
            oof_scores(items)
            for tau2 in taus2:
                _, M = sums_at(items, tau2, m2key)
                m = metrics_from_sums(M.sum(0))
                r = dict(tau_p=tau_p, gap=gap, tau2=float(tau2), F1=float(m["F1"]) if np.isfinite(m["F1"]) else 0.0, sens=float(m["sensibilité"]),
                         fa_h=float(m["fausses_alarmes_par_h"]), prec=float(m["précision"]))
                rows.append(r)
                if best is None or (r["F1"], -r["gap"], r["tau2"]) > (best["F1"], -best["gap"], best["tau2"]):
                    best, best_items = r, items
    return pd.DataFrame(rows), best, best_items


def main_tune(args):
    recs, _, _ = load_split(args.manifest, "val", args.m2key)
    t, best, items = tune(recs, args.m2key)
    # références à UN étage sur la même validation : la règle de la première évaluation, et la meilleure règle à un étage avec fusion libre
    p0 = json.load(open(args.params_stage1)) if os.path.exists(args.params_stage1) else dict(tau=0.8, n_min=1, merge_gap_s=6.0)
    _, M0 = per_patient_sums(recs, p0["tau"], p0["n_min"], args.m2key, 1, None, p0["merge_gap_s"])
    t1, b1 = tune_stage1(recs, args.m2key, np.round(np.arange(0.05, 0.96, 0.05), 2), (1, 2, 3), (1,), (6.0, 20.0, 40.0))
    _, M1 = per_patient_sums(recs, b1["tau"], b1["n_min"], args.m2key, 1, None, b1["merge_gap_s"])
    pats, M2 = sums_at(items, best["tau2"], args.m2key)
    model = fit_stage2(items)                                       # modèle final : entraîné sur TOUTE la validation
    cfg = dict(tau_p=best["tau_p"], gap=best["gap"], tau2=best["tau2"], m2key=args.m2key, val_F1_oof=best["F1"])
    os.makedirs(os.path.dirname(os.path.abspath(args.params)), exist_ok=True)
    json.dump(cfg, open(args.params, "w"), indent=2)
    pickle.dump(model, open(args.model, "wb"))
    X = np.concatenate([i["X"] for i in items]); y = np.concatenate([i["y"] for i in items])
    print("<<<COPY_FROM_HERE>>>")
    print(f"validation : {len(recs)} enregistrements, {len({r[0] for r in recs})} patients ; propositions retenues {len(y)} dont vraies {int(y.sum())} ({y.mean():.2f})")
    print(f"configuration choisie (scores hors pli, critère F1 par événement) : {cfg}")
    print("F1 maximal par (tau_p, fusion) :")
    print(t.groupby(["tau_p", "gap"]).F1.max().round(3).unstack().to_string())
    print("--- à un étage (validation, optimiste : la règle y est choisie) ---")
    for tag, M, extra in (("règle de la 1re évaluation", M0, f"tau={p0['tau']}, n_min={p0['n_min']}, fusion {p0['merge_gap_s']} s"),
                          ("meilleure règle à un étage, fusion libre", M1, f"tau={b1['tau']}, n_min={b1['n_min']}, fusion {b1['merge_gap_s']} s")):
        m = metrics_from_sums(M.sum(0))
        print(f"  {tag} ({extra}) : F1 {float(m['F1']):.3f} | sensibilité {float(m['sensibilité']):.3f} | fausses alarmes/h {float(m['fausses_alarmes_par_h']):.2f} | précision {float(m['précision']):.3f}")
    print("--- avec second étage (scores HORS PLI ; seul tau2 / tau_p / fusion sont choisis sur la validation) ---")
    print("\n".join(fmt_final(M2, 0, 0)))
    print("différence second étage - règle de la 1re évaluation (bootstrap par patient, apparié) :")
    print("\n".join(paired_diff(M2, M0)))
    print("différence second étage - meilleure règle à un étage avec fusion libre :")
    print("\n".join(paired_diff(M2, M1)))
    imp = []
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X, y, n_repeats=5, random_state=0, scoring="roc_auc")
        imp = sorted(zip(FEATURE_NAMES, r.importances_mean), key=lambda z: -z[1])[:6]
    except Exception:
        pass
    if imp:
        print("variables les plus utiles (importance par permutation, AUC) :", ", ".join(f"{n} {v:.3f}" for n, v in imp))
    print("<<<COPY_TO_HERE>>>")


def apply_stage2(recs, cfg, model, m2key):
    items = build(recs, cfg["tau_p"], cfg["gap"])
    score_items(model, items)
    return sums_at(items, cfg["tau2"], m2key)


def main_final(args):
    if not args.final_eval:
        raise SystemExit("ERREUR : la partition eval exige --final_eval (seconde évaluation finale, à déclarer comme telle).")
    if not (os.path.exists(args.params) and os.path.exists(args.model)):
        raise SystemExit("ERREUR : params_stage2.json / stage2.pkl introuvables : lancer d'abord `tune` sur la validation.")
    cfg = json.load(open(args.params))
    model = pickle.load(open(args.model, "rb"))
    m2key = args.m2key or cfg["m2key"]
    recs, lost_S, lost_E = load_split(args.manifest, "eval", m2key)
    pats, M = apply_stage2(recs, cfg, model, m2key)
    print("<<<COPY_FROM_HERE>>>")
    print(f"SECONDE évaluation sur eval, second étage FIGÉ : {cfg} ; M2 = {m2key}")
    print("\n".join(fmt_final(M, lost_S, lost_E)))
    print("\n".join(fmt_oracle(per_patient_oracle(recs, m2key))))
    if os.path.exists(args.params_stage1):
        p0 = json.load(open(args.params_stage1))
        _, M0 = per_patient_sums(recs, p0["tau"], p0["n_min"], m2key, 1, None, p0.get("merge_gap_s", MERGE_GAP_S))
        print(f"différence second étage - règle à un étage de la 1re évaluation (tau={p0['tau']}, n_min={p0['n_min']}) ; bootstrap par patient, apparié :")
        print("\n".join(paired_diff(M, M0)))
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test (numpy + scikit-learn)
def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    n_samp = 900 * FS
    st = np.arange(0, n_samp - WIN + 1, 500)
    t0 = st / FS

    def make_rec(seed, with_sz=True):
        r = np.random.default_rng(seed)
        p1 = np.clip(r.normal(0.05, 0.03, len(st)), 0, 1)
        S = []
        if with_sz:
            a = float(r.integers(100, 500))
            p1[(t0 >= a) & (t0 < a + 60)] = np.clip(r.normal(0.85, 0.05, int(((t0 >= a) & (t0 < a + 60)).sum())), 0, 1)     # vraie crise : longue, soutenue
            S.append((a, a + 60))
        for _ in range(3):                                                    # fausses alarmes : pics brefs (1-2 fenêtres) et moins hauts, loin des crises
            b = float(r.integers(20, 800))
            if not S or abs(b - S[0][0]) > 120:
                p1[(t0 >= b) & (t0 < b + 3)] = r.uniform(0.35, 0.6)
        return dict(starts=st, p1=p1, flat=np.zeros(len(st), bool), n_samples=n_samp, p2_x=np.tile([1.0, 0, 0], (len(st), 1))), S

    recs = []
    for k in range(80):
        sc, S = make_rec(k)
        recs.append((f"p{k:02d}", sc, S, [(S[0][0], S[0][1], 0)]))
    # 1) variables causales : modifier l'avenir d'une proposition ne change pas ses variables de contexte
    sc, S = make_rec(1)
    d = proposals(sc, 0.3, 6.0)
    f1 = features(sc, d)
    sc2 = dict(sc, p1=sc["p1"].copy())
    sc2["p1"][t0 > d[-1]["stop"] + 1] = 0.99                                  # on change tout ce qui suit la dernière proposition
    f2 = features(sc2, d)
    check("variables causales : le futur d'une proposition ne change aucune de ses 12 variables", np.allclose(f1[-1], f2[-1]) and f1.shape[1] == len(FEATURE_NAMES))
    check("la proposition suivante voit bien l'historique (compteur d'alarmes précédentes > 0 pour la 2e)", len(d) < 2 or features(sc, d)[1, FEATURE_NAMES.index("n_prop_10min")] >= 1)
    # 2) le second étage retire des fausses alarmes sans perdre les vraies crises (validation croisée par patient)
    items = build(recs, 0.3, 6.0)
    oof_scores(items)
    _, M_all = sums_at(items, 0.0, "x")                                       # tau2 = 0 : on garde tout (= étage 1 seul)
    f1s = {float(t): float(metrics_from_sums(sums_at(items, t, "x")[1].sum(0))["F1"]) for t in TAU2}
    tau_best = max(f1s, key=lambda t: (f1s[t], t))                            # tau2 choisi par le F1 hors pli, comme dans `tune`
    _, M_f = sums_at(items, tau_best, "x")
    a, b = metrics_from_sums(M_all.sum(0)), metrics_from_sums(M_f.sum(0))
    check(f"second étage (tau2 = {tau_best}) : fausses alarmes/h {float(a['fausses_alarmes_par_h']):.2f} -> {float(b['fausses_alarmes_par_h']):.2f} (baisse > 50 %), sensibilité "
          f"{float(a['sensibilité']):.2f} -> {float(b['sensibilité']):.2f} (perte <= 0,1)",
          b["fausses_alarmes_par_h"] < 0.5 * a["fausses_alarmes_par_h"] and b["sensibilité"] >= a["sensibilité"] - 0.1)
    # 3) hors pli : un patient n'est jamais noté par un modèle qui l'a vu (les plis séparent les patients)
    fold = patient_folds([i["pat"] for i in items])
    check("plis par patient : 5 plis non vides, chaque patient dans un seul pli", set(fold.values()) == set(range(5)) and len(fold) == 80)
    # 4) bootstrap apparié : différence nulle d'un résultat avec lui-même
    pa, Ma = sums_at(items, tau_best, "x")
    z = paired_diff(Ma, Ma)
    check("différence appariée d'un résultat avec lui-même = 0", all("+0.000  [+0.000, +0.000]" in l for l in z))
    # 5) final : refus sans --final_eval ; sans fichiers ; et sur des enregistrements synthétiques, la config figée donne les mêmes comptes que build+score
    try:
        main_final(argparse.Namespace(final_eval=False, params="x", model="y", m2key=None, manifest="z", params_stage1="w"))
        check("final sans --final_eval refusé", False)
    except SystemExit:
        check("final sans --final_eval refusé", True)
    try:
        main_final(argparse.Namespace(final_eval=True, params="/nonexistent.json", model="/nonexistent.pkl", m2key=None, manifest="z", params_stage1="w"))
        check("final sans params_stage2.json refusé", False)
    except SystemExit:
        check("final sans params_stage2.json refusé", True)
    model = fit_stage2(items)
    cfg = dict(tau_p=0.3, gap=6.0, tau2=tau_best)
    _, Mfin = apply_stage2(recs, cfg, model, "x")
    check("apply_stage2 : le modèle figé retire aussi des fausses alarmes sur ces enregistrements (>= 1 en moins qu'à tau2 = 0)",
          Mfin[:, 2].sum() < M_all[:, 2].sum())
    # 6) une seule classe dans l'entraînement : pas de plantage, toutes les propositions gardées
    only_pos = [dict(i, y=np.ones_like(i["y"])) for i in items[:3]]
    check("entraînement à une seule classe : pas de modèle, propositions toutes conservées", fit_stage2(only_pos) is None)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["tune", "final"])
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--manifest", default=base + "scores/scores_manifest.csv")
    ap.add_argument("--params", default=base + "scores/params_stage2.json")
    ap.add_argument("--model", default=base + "scores/stage2.pkl")
    ap.add_argument("--params_stage1", default=base + "scores/params.json", help="règle à un étage de la première évaluation (comparaison)")
    ap.add_argument("--m2key", default=None, help="tune : défaut windowglobal ; final : défaut = celui de params_stage2.json")
    ap.add_argument("--final_eval", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.cmd == "tune":
        a.m2key = a.m2key or "windowglobal"
        main_tune(a)
    elif a.cmd == "final":
        main_final(a)
    else:
        ap.error("donner tune ou final")
