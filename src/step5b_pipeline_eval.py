#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 5b -- Pipeline COMPLET sur ENREGISTREMENTS ENTIERS : détection (M1) + latéralisation (M2), évaluation par événement.
Partie numpy. Les scores par fenêtre (M1, M2) viennent de l'étape 5a. Deux commandes :

  tune  : règle de décision choisie sur les enregistrements de VALIDATION (role == val). Écrit params.json.
  final : applique la règle FIGÉE (params.json) aux enregistrements de role == eval, une seule fois, avec intervalles par patient.

Règle de décision (2 paramètres tunés + 1 fixe)
  * Fenêtre de 4 s tous les 2 s. Fenêtre « détectée » si p(M1) >= tau (les fenêtres plates ne sont jamais détectées).
  * Fenêtres détectées séparées de <= 6 s (fixe) -> une même détection ; gardée si elle compte >= n_min fenêtres.
  * Label d'une détection = classe de plus forte probabilité MOYENNE de M2 sur ses fenêtres.
Appariement (par DÉBUT de détection, pour qu'une détection géante ne puisse pas « couvrir » tout l'enregistrement) :
  une détection est vraie si son début tombe entre (début de crise - 30 s) et la fin d'une crise annotée de N'IMPORTE QUEL type, sinon fausse alarme.
  Une crise annotée (intervalles fusionnés, tous types) est détectée si au moins une détection vraie pour elle existe.
  Un événement lateralisable (règle de l'étape 1) prend le label de la détection éligible qui le chevauche le plus ; sans détection : MANQUÉ.
  La part du temps en alarme est rapportée (garde-fou contre les détections qui couvrent tout).
Critère de choix sur la validation : F1 par événement = 2 TP / (2 TP + FN + FP).
Mesures : sensibilité, fausses alarmes par heure, précision, F1 ; latéralisation de bout en bout (un événement manqué compte comme faux)
  et conditionnelle (parmi les événements détectés) ; intervalles de confiance = bootstrap par patient.
"""
import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_inventaire_labels import parse_annotation, extract_events, label_of, REF
from step2b_cache_windows import seizure_intervals, FS, WIN, IDX

MERGE_GAP_S = 6.0
PRE_TOL_S = 30.0      # une détection compte pour une crise si elle COMMENCE entre (début - 30 s) et la fin de la crise
COLS = ["tp_ref", "fn", "fp", "tp_det", "hours", "n_L", "n_R", "n_B", "c_L", "c_R", "c_B", "d_L", "d_R", "d_B", "alarm_h"]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def detect(starts, p1, flat, tau, n_min, merge_gap=MERGE_GAP_S):
    on = np.where((p1 >= tau) & ~flat)[0]
    if len(on) == 0:
        return []
    s = starts[on] / FS
    e = s + WIN / FS
    groups, cur = [], [0]
    for k in range(1, len(on)):
        if s[k] <= e[cur[-1]] + merge_gap:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    return [dict(start=float(s[g[0]]), stop=float(e[g[-1]]), idx=on[g]) for g in groups if len(g) >= n_min]


def reference_of(rows):
    """Crises annotées (tous types, fusionnées) et événements lateralisables avec leur label (0/1/2)."""
    S = seizure_intervals(rows)
    E = []
    for ev in extract_events(rows):
        lab = label_of(ev, REF)
        if lab is not None:
            E.append((ev["start"], ev["stop"], IDX[lab]))
    return S, E


def smooth_p1(p1, w):
    """Moyenne glissante centrée sur w fenêtres (w impair ; w = 1 : aucun lissage). Aux bords, moyenne sur les fenêtres existantes."""
    if w <= 1:
        return p1
    k, h, n = np.ones(w), w // 2, len(p1)          # « full » puis découpe : np.convolve(..., "same") rend max(n, w) valeurs si n < w
    return np.convolve(p1, k, "full")[h:h + n] / np.convolve(np.ones(n), k, "full")[h:h + n]


def eval_recording(sc, S, E, tau, n_min, m2key, smooth=1, k_lat=None, gap=MERGE_GAP_S):
    """smooth : lissage temporel de p(M1) avant le seuil. k_lat : le label d'une détection vient des k PREMIÈRES fenêtres détectées
    (None = toutes), car l'étiquette est définie sur le début de la crise. gap : fusion des détections séparées de <= gap secondes."""
    dets = detect(sc["starts"], smooth_p1(sc["p1"], smooth), sc["flat"], tau, n_min, gap)
    return eval_dets(sc, S, E, dets, m2key, k_lat)


def eval_dets(sc, S, E, dets, m2key, k_lat=None):
    """Appariement et comptes pour une liste de détections déjà construite (dict start, stop, idx) : réutilisé par le second étage (étape 9)."""
    for d in dets:
        ii = d["idx"] if k_lat is None else d["idx"][:k_lat]
        d["label"] = int(sc["p2_" + m2key][ii].mean(0).argmax())
    def hit(d, a, b):
        return a - PRE_TOL_S <= d["start"] <= b
    tp_det = sum(1 for d in dets if any(hit(d, a, b) for a, b in S))
    fp = len(dets) - tp_det
    tp_ref = sum(1 for a, b in S if any(hit(d, a, b) for d in dets))
    n = np.zeros(3); c = np.zeros(3); d_ = np.zeros(3)
    for a, b, y in E:
        n[y] += 1
        best, lab = 0.0, -1
        for d in dets:
            o = overlap(d["start"], d["stop"], a, b)
            if hit(d, a, b) and o > best:
                best, lab = o, d["label"]
        if lab >= 0:
            d_[y] += 1
            c[y] += int(lab == y)
    hours = sc["n_samples"] / FS / 3600
    alarm_h = sum(d["stop"] - d["start"] for d in dets) / 3600
    return np.array([tp_ref, len(S) - tp_ref, fp, tp_det, hours, *n, *c, *d_, alarm_h], float)


def metrics_from_sums(v):
    v = np.asarray(v, float)
    tp, fn, fp, tpd, h, nL, nR, nB, cL, cR, cB, dL, dR, dB, ah = np.moveaxis(v, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # classes absentes -> NaN, ignorées des moyennes
        e2e = np.stack([cL / nL, cR / nR, cB / nB], axis=-1)
        cond = np.stack([cL / dL, cR / dR, cB / dB], axis=-1)
        return dict(sensibilité=tp / (tp + fn), fausses_alarmes_par_h=fp / h, précision=tpd / (tpd + fp), part_du_temps_en_alarme=ah / h, F1=2 * tp / (2 * tp + fn + fp),
                    exactitude_bout_en_bout=(cL + cR + cB) / (nL + nR + nB), équilibrée_bout_en_bout=np.nanmean(e2e, axis=-1),
                    rappel_G=e2e[..., 0], rappel_D=e2e[..., 1], rappel_B=e2e[..., 2],
                    exactitude_conditionnelle=(cL + cR + cB) / (dL + dR + dB), équilibrée_conditionnelle=np.nanmean(cond, axis=-1))


def load_split(manifest, role, m2key, scores_dir=None):
    """Retourne (enregistrements traités [(patient, scores, S, E)], nb de crises / événements dans les enregistrements NON traités)."""
    m = pd.read_csv(manifest)
    m = m[m["role"] == role]
    recs, lost_S, lost_E = [], 0, 0
    for r in m.itertuples():
        _, rows = parse_annotation(r.csv_path)
        S, E = reference_of(rows)
        if r.status != "ok":
            lost_S += len(S)
            lost_E += len(E)
            continue
        z = np.load(r.score_path)
        sc = dict(starts=z["starts"], p1=z["p1"], flat=z["flat"], n_samples=int(z["n_samples"]), **{"p2_" + m2key: z["p2_" + m2key]})
        recs.append((r.patient, sc, S, E))
    return recs, lost_S, lost_E


def per_patient_sums(recs, tau, n_min, m2key, smooth=1, k_lat=None, gap=MERGE_GAP_S):
    acc = {}
    for pat, sc, S, E in recs:
        acc[pat] = acc.get(pat, 0) + eval_recording(sc, S, E, tau, n_min, m2key, smooth, k_lat, gap)
    pats = sorted(acc)
    return pats, np.array([acc[p] for p in pats])


def tune(recs, m2key, taus, n_mins, smooths=(1,), gaps=(MERGE_GAP_S,)):
    rows = []
    for gap in gaps:
        for sm in smooths:
            for n_min in n_mins:
                for tau in taus:
                    _, M = per_patient_sums(recs, tau, n_min, m2key, sm, None, gap)
                    m = metrics_from_sums(M.sum(0))
                    rows.append(dict(tau=float(tau), n_min=int(n_min), smooth=int(sm), gap=float(gap), F1=float(m["F1"]), sens=float(m["sensibilité"]),
                                     fa_h=float(m["fausses_alarmes_par_h"]), prec=float(m["précision"]),
                                     alarm=float(m["part_du_temps_en_alarme"])))
    t = pd.DataFrame(rows).fillna({"F1": 0.0})
    # égalité : le moins de lissage, la fusion la plus courte, puis le tau le plus haut
    best = t.sort_values(["F1", "smooth", "gap", "tau"], ascending=[False, True, True, False]).iloc[0]
    return t, dict(tau=float(best.tau), n_min=int(best.n_min), smooth=int(best.smooth), merge_gap_s=float(best.gap), val_F1=float(best.F1))


def select_k_lat(recs, m2key, ks=(None, 1, 2, 3, 4, 6)):
    """Choisit k (nb de premières fenêtres utilisées pour le label) par l'exactitude équilibrée en détection PARFAITE sur la validation
    (indépendant de la détection). Égalité : on garde « toutes les fenêtres »."""
    rows = []
    for k in ks:
        acc = {}
        for pat, sc, S, E in recs:
            acc[pat] = acc.get(pat, 0) + oracle_counts(sc, E, m2key, k)
        Mo = np.array([acc[p] for p in sorted(acc)])
        m = oracle_metrics(Mo.sum(0))
        rows.append(dict(k=("toutes" if k is None else k), bal=float(m["équilibrée_oracle"]), acc=float(m["exactitude_oracle"]),
                         G=float(m["rappel_G_oracle"]), D=float(m["rappel_D_oracle"]), B=float(m["rappel_B_oracle"])))
    t = pd.DataFrame(rows)
    best_i = int(np.argmax(t.bal.round(6).values))        # égalité : premier de la liste, donc « toutes les fenêtres » (le plus simple)
    return t, ks[best_i]


def bootstrap(M, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(M), (n_boot, len(M)))
    return metrics_from_sums(M[idx].sum(1))


ORACLE_COLS = ["n_L", "n_R", "n_B", "c_L", "c_R", "c_B"]


def oracle_counts(sc, E, m2key, k_lat=None):
    """Latéralisation avec DÉTECTION PARFAITE : pour chaque événement latéralisable, on prend les fenêtres non plates
    entièrement contenues dans l'événement, on moyenne les probabilités de M2 et on prend l'argmax.
    Un événement sans aucune fenêtre utilisable n'est pas compté (ni dans n, ni dans c)."""
    n = np.zeros(3); c = np.zeros(3)
    t0 = sc["starts"] / FS
    t1 = t0 + WIN / FS
    for a, b, y in E:
        idx = np.where((t0 >= a) & (t1 <= b) & ~sc["flat"])[0]
        if len(idx) == 0:
            continue
        n[y] += 1
        ii = idx if k_lat is None else idx[:k_lat]
        c[y] += int(sc["p2_" + m2key][ii].mean(0).argmax() == y)
    return np.array([*n, *c], float)


def per_patient_oracle(recs, m2key, k_lat=None):
    acc = {}
    for pat, sc, S, E in recs:
        acc[pat] = acc.get(pat, 0) + oracle_counts(sc, E, m2key, k_lat)
    return np.array([acc[p] for p in sorted(acc)])


def oracle_metrics(v):
    v = np.asarray(v, float)
    nL, nR, nB, cL, cR, cB = np.moveaxis(v, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.stack([cL / nL, cR / nR, cB / nB], axis=-1)
        return dict(exactitude_oracle=(cL + cR + cB) / (nL + nR + nB), équilibrée_oracle=np.nanmean(r, axis=-1),
                    rappel_G_oracle=r[..., 0], rappel_D_oracle=r[..., 1], rappel_B_oracle=r[..., 2])


def fmt_oracle(Mo, n_boot=2000):
    point = oracle_metrics(Mo.sum(0))
    rng = np.random.default_rng(0)
    boot = oracle_metrics(Mo[rng.integers(0, len(Mo), (n_boot, len(Mo)))].sum(1))
    L = [f"latéralisation avec détection PARFAITE (fenêtres non plates dans l'événement) : événements G/D/B = "
         f"{int(Mo[:, 0].sum())}/{int(Mo[:, 1].sum())}/{int(Mo[:, 2].sum())}"]
    for k, v in point.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        L.append(f"  {k:30s} {float(v):.3f}  [{lo:.3f}, {hi:.3f}]")
    return L


def fmt_final(M, lost_S, lost_E, n_boot=2000):
    point = metrics_from_sums(M.sum(0))
    boot = bootstrap(M, n_boot)
    L = [f"{len(M)} patients ; {M[:, COLS.index('hours')].sum():.1f} h d'enregistrement ; crises annotées (tous types) {int(M[:, 0].sum() + M[:, 1].sum())} ; "
         f"événements latéralisables G/D/B {int(M[:, 5].sum())}/{int(M[:, 6].sum())}/{int(M[:, 7].sum())}"]
    for k, v in point.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        L.append(f"  {k:30s} {float(v):.3f}  [{lo:.3f}, {hi:.3f}]")
    tp, fn = M[:, 0].sum(), M[:, 1].sum()
    if lost_S:
        L.append(f"  enregistrements NON traités (rejetés) : {lost_S} crises et {lost_E} événements latéralisables perdus ; "
                 f"sensibilité si on les compte comme manqués : {tp / (tp + fn + lost_S):.3f}")
    return L


def main(args):
    if args.cmd == "tune":
        m2key = args.m2key or "windowglobal"
        recs, _, _ = load_split(args.manifest, "val", m2key)
        taus = np.round(np.r_[0.01, 0.02, 0.03, np.arange(0.05, 0.96, 0.05)], 2)
        n_mins = (1, 2, 3, 4, 6, 8)
        smooths = tuple(int(x) for x in str(getattr(args, "smooths", "1")).split(","))       # défaut : pas de lissage (il n'a jamais aidé pour M1)
        gaps = tuple(float(x) for x in str(getattr(args, "gaps", "6,12,20,40")).split(","))   # fusion des détections proches (secondes)
        t, best = tune(recs, m2key, taus, n_mins, smooths, gaps)
        tk, k_best = select_k_lat(recs, m2key)
        best["k_lat"] = k_best
        os.makedirs(os.path.dirname(os.path.abspath(args.params)), exist_ok=True)
        with open(args.params, "w") as f:
            json.dump(dict(best, m2key=m2key), f, indent=2)
        print("<<<COPY_FROM_HERE>>>")
        print(f"enregistrements de validation traités : {len(recs)} ; patients {len({r[0] for r in recs})}")
        print(f"règle choisie (critère F1 par événement) : {best}")
        print("F1 maximal selon n_min (sur tous les seuils) :", t.groupby("n_min").F1.max().round(3).to_dict())
        edge_n, edge_t = best["n_min"] == max(n_mins), best["tau"] in (float(taus.min()), float(taus.max()))
        edge_s = len(smooths) > 1 and best["smooth"] == max(smooths)
        edge_g = best["merge_gap_s"] == max(gaps)
        print("ATTENTION : l'optimum est au bord de la grille (" + ("n_min " if edge_n else "") + ("tau " if edge_t else "") + ("lissage " if edge_s else "") + ("fusion" if edge_g else "") + ") : élargir la grille."
              if (edge_n or edge_t or edge_s or edge_g) else "l'optimum n'est pas au bord de la grille (tau, n_min, lissage, fusion).")
        print("F1 maximal selon la fusion des détections (s) :", t.groupby("gap").F1.max().round(3).to_dict())
        print("--- latéralisation, détection PARFAITE : combien de premières fenêtres utiliser pour le label (validation) ---")
        print(tk.round(3).to_string(index=False))
        print("k choisi (exactitude équilibrée maximale) :", "toutes" if k_best is None else k_best)
        print("meilleur tau pour chaque n_min :", {int(k): float(v) for k, v in t.loc[t.groupby("n_min").F1.idxmax()].set_index("n_min").tau.items()})
        print(t[(t.n_min == best["n_min"]) & (t.smooth == best["smooth"]) & (t.gap == best["merge_gap_s"])].round(3).to_string(index=False))
        _, Mv = per_patient_sums(recs, best["tau"], best["n_min"], m2key, best["smooth"], k_best, best["merge_gap_s"])
        print("--- avec la règle choisie, sur la validation (OPTIMISTE : la règle y a été choisie) ---")
        print("\n".join(fmt_final(Mv, 0, 0)))
        print("\n".join(fmt_oracle(per_patient_oracle(recs, m2key, k_best))))
        print("<<<COPY_TO_HERE>>>")
    else:
        if not os.path.exists(args.params):
            raise SystemExit("params.json introuvable : lancer d'abord `tune` sur la validation.")
        with open(args.params) as f:
            P = json.load(f)
        m2key = args.m2key or P["m2key"]                       # --m2key : M2 comparateur, MÊME règle de détection figée
        recs, lost_S, lost_E = load_split(args.manifest, "eval", m2key)
        sm, kl = P.get("smooth", 1), P.get("k_lat")
        pats, M = per_patient_sums(recs, P["tau"], P["n_min"], m2key, sm, kl, P.get("merge_gap_s", MERGE_GAP_S))
        print("<<<COPY_FROM_HERE>>>")
        print(f"règle FIGÉE : tau={P['tau']}, n_min={P['n_min']}, lissage={P.get('smooth', 1)}, k_lat={P.get('k_lat')}, fusion {P['merge_gap_s']} s, M2 = {m2key} (règle choisie sur la validation avec {P['m2key']}, F1 = {P['val_F1']:.3f})")
        print("\n".join(fmt_final(M, lost_S, lost_E)))
        print("\n".join(fmt_oracle(per_patient_oracle(recs, m2key, kl))))
        print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test
def selftest():
    import tempfile, shutil
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    # detect : fusion et n_min
    starts = np.arange(0, 100 * 500, 500)
    p1 = np.zeros(100)
    p1[10:14] = 0.9              # 4 fenêtres consécutives
    p1[16] = 0.9                 # isolée mais séparée de 6 s ou moins du groupe précédent -> fusion
    p1[40] = 0.9                 # isolée, loin
    flat = np.zeros(100, bool)
    d1 = detect(starts, p1, flat, 0.5, 1)
    check(f"detect : 2 détections (fusion à <= 6 s, groupe isolé séparé) : {len(d1)}", len(d1) == 2 and len(d1[0]["idx"]) == 5)
    check("detect : n_min=2 supprime la détection d'une seule fenêtre", len(detect(starts, p1, flat, 0.5, 2)) == 1)
    fl = flat.copy(); fl[10:14] = True
    check("detect : les fenêtres plates ne sont jamais détectées", all(d["start"] > 30 for d in detect(starts, p1, fl, 0.5, 1)))
    # eval_recording sur un enregistrement synthétique : 2 crises annotées (gauche, bilatéral), 1 détectée + 1 fausse alarme
    n_samp = 600 * FS
    st = np.arange(0, n_samp - WIN + 1, 500)
    p1 = np.full(len(st), 0.05)
    t0 = st / FS
    p1[(t0 >= 100) & (t0 < 130)] = 0.9            # crise 1 [100, 130] détectée
    p1[(t0 >= 400) & (t0 < 410)] = 0.9            # fausse alarme
    p2 = np.zeros((len(st), 3)); p2[:, 0] = 1.0   # M2 répond toujours « gauche »
    sc = dict(starts=st, p1=p1, flat=np.zeros(len(st), bool), n_samples=n_samp, p2_x=p2)
    rows = [("FP1-F7", 100.0, 130.0, "fnsz"), ("F7-T3", 100.0, 130.0, "fnsz"),           # événement gauche
            ("FP1-F7", 300.0, 330.0, "fnsz"), ("F7-T3", 300.0, 330.0, "fnsz"), ("FP2-F8", 300.0, 330.0, "fnsz"), ("F8-T4", 300.0, 330.0, "fnsz")]  # bilatéral, manqué
    S, E = reference_of(rows)
    check("reference_of : 2 crises, événements de labels gauche (0) et bilatéral (2)", len(S) == 2 and sorted(e[2] for e in E) == [0, 2])
    v = eval_recording(sc, S, E, 0.5, 1, "x")
    d = dict(zip(COLS, v))
    check(f"comptes : 1 crise détectée, 1 manquée, 1 fausse alarme ({d['tp_ref']:.0f}/{d['fn']:.0f}/{d['fp']:.0f})", (d["tp_ref"], d["fn"], d["fp"]) == (1, 1, 1))
    check("latéralisation : gauche juste (1/1), bilatéral manqué (0 détecté)", (d["n_L"], d["c_L"], d["d_L"], d["n_B"], d["c_B"], d["d_B"]) == (1, 1, 1, 1, 0, 0))
    m = metrics_from_sums(v)
    check(f"métriques : sensibilité 0,5 ; F1 = 2/(2+1+1) = 0,5 ; équilibrée de bout en bout (1 + 0)/2 = 0,5 ; conditionnelle 1,0",
          abs(m["sensibilité"] - 0.5) < 1e-9 and abs(m["F1"] - 0.5) < 1e-9 and abs(m["équilibrée_bout_en_bout"] - 0.5) < 1e-9
          and abs(m["équilibrée_conditionnelle"] - 1.0) < 1e-9 and abs(m["fausses_alarmes_par_h"] - 6.0) < 1e-6)
    # une détection géante (tout l'enregistrement) ne doit PAS compter pour une crise qui commence loin de son début
    sc_all = dict(sc, p1=np.full(len(st), 0.9))
    va = dict(zip(COLS, eval_recording(sc_all, S, E, 0.5, 1, "x")))
    check(f"détection géante : 0 crise détectée, 1 fausse alarme, 100 % du temps en alarme ({va['tp_ref']:.0f}/{va['fp']:.0f}/{va['alarm_h'] * 3600:.0f} s)",
          va["tp_ref"] == 0 and va["fp"] == 1 and abs(va["alarm_h"] * 3600 - 600) < 5)
    # détection qui commence 20 s AVANT la crise : comptée (tolérance 30 s)
    p1_pre = np.full(len(st), 0.05); p1_pre[(t0 >= 80) & (t0 < 130)] = 0.9
    vp = dict(zip(COLS, eval_recording(dict(sc, p1=p1_pre), S, E, 0.5, 1, "x")))
    check("détection commençant 20 s avant la crise : comptée (tolérance 30 s)", vp["tp_ref"] == 1 and vp["fp"] == 0)
    # fusion : deux détections séparées de 20 s fusionnent avec gap=40 mais pas avec gap=6
    pg = np.zeros(100); pg[10:12] = 0.9; pg[22:24] = 0.9                   # trou d'environ 20 s entre les deux
    flg = np.zeros(100, bool)
    check("fusion : 2 détections avec gap=6, 1 avec gap=40", len(detect(starts, pg, flg, 0.5, 1, 6.0)) == 2 and len(detect(starts, pg, flg, 0.5, 1, 40.0)) == 1)
    # lissage : un pic isolé (1 fenêtre) tombe sous le seuil après moyenne sur 5 fenêtres
    pk = np.zeros(20); pk[10] = 0.9
    check("lissage : un pic isolé de 0,9 devient 0,18 avec w=5, et w=1 ne change rien", abs(smooth_p1(pk, 5)[10] - 0.18) < 1e-9 and np.allclose(smooth_p1(pk, 1), pk))
    check("lissage : tableau plus court que la fenêtre (1 ou 2 valeurs) -> même longueur, valeurs inchangées",
          len(smooth_p1(np.array([0.7]), 7)) == 1 and np.allclose(smooth_p1(np.array([0.7]), 7), 0.7) and len(smooth_p1(np.array([0.2, 0.4]), 5)) == 2)
    # k_lat : le label vient des k premières fenêtres de la détection
    p2k = np.zeros((len(st), 3)); p2k[:, 2] = 1.0; p2k[(t0 >= 100) & (t0 < 106), 0] = 2.0     # début de crise : gauche ; ensuite : bilatéral
    sck = dict(sc, p2_x=p2k)
    lab_all = eval_recording(sck, S, E, 0.5, 1, "x", 1, None); lab_k = eval_recording(sck, S, E, 0.5, 1, "x", 1, 2)
    dk_all, dk_k = dict(zip(COLS, lab_all)), dict(zip(COLS, lab_k))
    check("k_lat : toutes les fenêtres -> bilatéral (gauche faux) ; 2 premières -> gauche (juste)", dk_all["c_L"] == 0 and dk_k["c_L"] == 1)
    ok_k = oracle_counts(sck, E, "x", 2); ok_all = oracle_counts(sck, E, "x", None)
    check("oracle avec k : même comportement que ci-dessus", ok_k[3] == 1 and ok_all[3] == 0)
    tk_, kb = select_k_lat([("p", sck, S, E)], "x", ks=(None, 1, 2))
    check(f"select_k_lat choisit un k fini ({kb}) quand le début est plus informatif", kb in (1, 2))
    # oracle : événement gauche prédit gauche (juste), événement bilatéral prédit gauche (faux) ; fenêtres plates ignorées
    p2o = np.zeros((len(st), 3)); p2o[:, 0] = 1.0
    sc_o = dict(starts=st, p1=p1, flat=np.zeros(len(st), bool), n_samples=n_samp, p2_x=p2o)
    vo = oracle_counts(sc_o, E, "x")
    mo = oracle_metrics(vo)
    check(f"oracle : gauche juste, bilatéral faux -> rappels 1/0, équilibrée 0,5 ({vo})",
          list(vo) == [1, 0, 1, 1, 0, 0] and abs(mo["équilibrée_oracle"] - 0.5) < 1e-9 and mo["rappel_G_oracle"] == 1.0 and mo["rappel_B_oracle"] == 0.0)
    fl_o = np.zeros(len(st), bool); fl_o[(t0 >= 95) & (t0 < 135)] = True       # toutes les fenêtres de l'événement gauche sont plates
    vo2 = oracle_counts(dict(sc_o, flat=fl_o), E, "x")
    check("oracle : un événement dont toutes les fenêtres sont plates n'est pas compté", vo2[0] == 0 and vo2[2] == 1)
    # tune + final sur un manifeste synthétique (2 patients de validation, 2 d'évaluation, 1 enregistrement rejeté)
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        rows_m = []
        for i, (role, pat) in enumerate([("val", "pa"), ("val", "pb"), ("eval", "pc"), ("eval", "pd"), ("eval", "pe")]):
            stem = f"{pat}000000_s001_t000"
            csv = os.path.join(tmp, stem + ".csv")
            with open(csv, "w") as f:
                f.write("# duration = 600 secs\nchannel,start_time,stop_time,label,confidence\n")
                for ch, a, b, lab in [("FP1-F7", 100, 130, "fnsz"), ("F7-T3", 100, 130, "fnsz")]:
                    f.write(f"{ch},{a},{b},{lab},1.0\n")
            if pat == "pe":
                rows_m.append(dict(role=role, recording=stem, patient=pat, status="rejected", csv_path=csv, score_path=""))
                continue
            pp = np.clip(rng.normal(0.1, 0.05, len(st)), 0, 1)
            pp[(t0 >= 100) & (t0 < 130)] = 0.8
            p2s = np.zeros((len(st), 3)); p2s[:, 0] = 1.0
            sp = os.path.join(tmp, stem + ".npz")
            np.savez(sp, starts=st, p1=pp, flat=np.zeros(len(st), bool), n_samples=n_samp, p2_x=p2s)
            rows_m.append(dict(role=role, recording=stem, patient=pat, status="ok", csv_path=csv, score_path=sp))
        man = os.path.join(tmp, "manifest.csv")
        pd.DataFrame(rows_m).to_csv(man, index=False)
        params = os.path.join(tmp, "params.json")
        args = argparse.Namespace(cmd="tune", manifest=man, m2key="x", params=params)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(args)
        tune_out = buf.getvalue()
        P = json.load(open(params))
        check(f"tune : règle trouvée (tau={P['tau']}, n_min={P['n_min']}) avec F1 = {P['val_F1']:.2f} > 0,9 sur un jeu facile", P["val_F1"] > 0.9 and 0.15 <= P["tau"] <= 0.8)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(argparse.Namespace(cmd="final", manifest=man, m2key="x", params=params))
        out = buf.getvalue()
        check("tune : le rapport donne la latéralisation de bout en bout sur la validation", "exactitude_bout_en_bout" in tune_out and "F1 maximal selon n_min" in tune_out)
        check("final : appliqué à role == eval seulement (2 patients traités) et enregistrement rejeté signalé",
              "2 patients" in out and "NON traités" in out and "1 crises" in out)
        lost_ok = "sensibilité si on les compte comme manqués : 0.667" in out
        check("final : sensibilité en comptant la crise du rejeté comme manquée = 2/3", lost_ok)
        os.remove(params)
        try:
            main(argparse.Namespace(cmd="final", manifest=man, m2key="x", params=params))
            check("final sans params.json refusé", False)
        except SystemExit:
            check("final sans params.json refusé (arrêt avec erreur)", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # bootstrap : l'intervalle contient l'estimation
    M = np.array([[3, 1, 2, 3, 1.0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0.1], [2, 2, 0, 2, 1.0, 1, 0, 2, 1, 0, 1, 1, 0, 2, 0.1], [4, 0, 1, 4, 2.0, 2, 1, 1, 2, 1, 1, 2, 1, 1, 0.2]], float)
    est = metrics_from_sums(M.sum(0))["F1"]
    lo, hi = np.nanpercentile(bootstrap(M, 1000)["F1"], [2.5, 97.5])
    check(f"bootstrap par patient : IC [{lo:.2f}, {hi:.2f}] contient l'estimation {est:.2f}", lo <= est <= hi)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["tune", "final"])
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--manifest", default=base + "scores/scores_manifest.csv")
    ap.add_argument("--params", default=base + "scores/params_v2.json")
    ap.add_argument("--m2key", default=None, help="nom de l'ensemble M2 (défini à l'étape 5a). tune : défaut windowglobal ; final : défaut = celui de params.json")
    ap.add_argument("--smooths", default="1", help="tune : largeurs de lissage essayées (fenêtres), ex. 1,3,5")
    ap.add_argument("--gaps", default="6,12,20,40", help="tune : fusions de détections essayées (secondes)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.cmd:
        ap.error("donner tune ou final")
    main(a)
