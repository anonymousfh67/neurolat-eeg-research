#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 27 -- (1) Matrice de confusion hors pli de B_asym80 ; (2) re-notation des prédictions FIGÉES du pipeline complet (partition eval) selon
chaque définition alternative du label (équivalent de la figure 5 de l'article, refait avec le pipeline actuel).

Aucun entraînement, aucun réglage : les scores par fenêtre de l'étape 5a et la règle figée (scores/params*.json) sont relus tels quels. CPU, quelques minutes.
ATTENTION : `rescore` relit les scores de la partition eval (déjà calculés). Ce n'est pas un nouveau choix, mais c'est un regard de plus à déclarer.

Commandes
  confusion : matrice de confusion par événement des scores hors pli (lr_cv/B_asym80_val_scores.csv), 230 patients, 3 156 événements.
  rescore   : prédiction de bout en bout de chaque événement de eval (même appariement que l'étape 5b : détection vraie si son début est entre
              (début de crise - 30 s) et la fin ; label = détection éligible qui recouvre le plus l'événement ; sans détection : MANQUÉ, compté faux),
              puis, pour chaque variante de label de l'étape 25 (labels_variantes_eval.csv) : exactitude, exactitude équilibrée, taux de la classe
              majoritaire, écart exactitude - majoritaire, avec IC95 bootstrap par patient. Contrôle : la ligne « référence » doit redonner le rapport 5b
              (exactitude 0,398 ; équilibrée 0,435 pour les 392 événements des enregistrements traités).

Usage
  python step27_rescoring_confusion.py confusion --scores .../lr_cv/B_asym80_val_scores.csv
  python step27_rescoring_confusion.py rescore --manifest .../scores/scores_manifest.csv --params .../scores/params_v2.json \
         --labels .../label_sensitivity/labels_variantes_eval.csv --outdir .../label_sensitivity
  python step27_rescoring_confusion.py --selftest
"""
import argparse, json, os, sys, tempfile, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = ("left", "right", "bilateral")
CIDX = {c: i for i, c in enumerate(CLASSES)}


# ----------------------------------------------------------------------------------------------------------------------
# 1. Matrice de confusion hors pli
# ----------------------------------------------------------------------------------------------------------------------
def confusion_report(scores_csv):
    d = pd.read_csv(scores_csv)
    g = d.groupby(["recording", "eid"]).agg(patient=("patient", "first"), y=("y", "first"), pL=("pL", "mean"), pR=("pR", "mean"), pB=("pB", "mean")).reset_index()
    y = g["y"].astype(int).values
    pred = g[["pL", "pR", "pB"]].values.argmax(1)
    C = np.zeros((3, 3), int)
    np.add.at(C, (y, pred), 1)
    n = C.sum(1)
    rec = np.diag(C) / np.maximum(n, 1)
    L = [f"{len(g)} événements, {g.patient.nunique()} patients (hors pli)",
         "matrice de confusion (lignes : label de la règle ; colonnes : prédit) -- effectifs puis % de la ligne"]
    for i, c in enumerate(CLASSES):
        L.append(f"  {c:10s} " + " ".join(f"{C[i, j]:5d}" for j in range(3)) + "   |   " + " ".join(f"{100 * C[i, j] / max(n[i], 1):5.1f}" for j in range(3)) + f"   (n = {n[i]})")
    L.append(f"rappels G/D/B : {rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f} ; exactitude équilibrée {rec.mean():.3f} ; exactitude {np.trace(C) / C.sum():.3f}")
    L.append(f"confusions gauche <-> droite : {C[0, 1] + C[1, 0]} sur {n[0] + n[1]} événements unilatéraux")
    return L, C


# ----------------------------------------------------------------------------------------------------------------------
# 2. Re-notation des prédictions figées
# ----------------------------------------------------------------------------------------------------------------------
def predict_events(sc, S, spans, tau, n_min, m2key, smooth=1, k_lat=None, gap=6.0):
    """Label prédit (0/1/2) pour chaque événement (a, b) d'un enregistrement, ou -1 si manqué. Même logique que step5b.eval_dets."""
    from step5b_pipeline_eval import detect, smooth_p1, overlap, PRE_TOL_S
    dets = detect(sc["starts"], smooth_p1(sc["p1"], smooth), sc["flat"], tau, n_min, gap)
    for d in dets:
        ii = d["idx"] if k_lat is None else d["idx"][:k_lat]
        d["label"] = int(sc["p2_" + m2key][ii].mean(0).argmax())
    out = []
    for a, b in spans:
        best, lab = 0.0, -1
        for d in dets:
            o = overlap(d["start"], d["stop"], a, b)
            if (a - PRE_TOL_S <= d["start"] <= b) and o > best:
                best, lab = o, d["label"]
        out.append(lab)
    return out


def build_predictions(manifest, params, m2key=None):
    from step1_inventaire_labels import parse_annotation, extract_events, onset_counts, REF
    from step2b_cache_windows import seizure_intervals
    with open(params) as f:
        P = json.load(f)
    m2key = m2key or P["m2key"]
    m = pd.read_csv(manifest)
    m = m[m["role"] == "eval"]
    rows_out = []
    for r in m.itertuples():
        _, rows = parse_annotation(r.csv_path)
        evs = extract_events(rows)
        stem = os.path.basename(r.csv_path)[:-4]
        if r.status != "ok":
            for i, e in enumerate(evs):
                rows_out.append(dict(recording=stem, event_idx=i, patient=r.patient, processed=False, pred=-1, focal=bool(onset_counts(e, REF["W"], REF["count"])[0])))
            continue
        z = np.load(r.score_path)
        sc = dict(starts=z["starts"], p1=z["p1"], flat=z["flat"], n_samples=int(z["n_samples"]), **{"p2_" + m2key: z["p2_" + m2key]})
        S = seizure_intervals(rows)
        preds = predict_events(sc, S, [(e["start"], e["stop"]) for e in evs], P["tau"], P["n_min"], m2key, P.get("smooth", 1), P.get("k_lat"), P.get("merge_gap_s", 6.0))
        for i, (e, p) in enumerate(zip(evs, preds)):
            rows_out.append(dict(recording=stem, event_idx=i, patient=r.patient, processed=True, pred=int(p), focal=bool(onset_counts(e, REF["W"], REF["count"])[0])))
    return pd.DataFrame(rows_out), P, m2key


def _counts(lab_idx, pred, patients):
    """Par patient : effectifs par classe (3), corrects par classe (3), détectés (1), corrects parmi les détectés (1)."""
    pc, up = pd.factorize(pd.Series(patients))
    A = np.zeros((len(up), 8))
    for c in range(3):
        m = lab_idx == c
        np.add.at(A[:, c], pc[m], 1)
        np.add.at(A[:, 3 + c], pc[m], (pred[m] == c).astype(float))
    det = pred >= 0
    np.add.at(A[:, 6], pc[det], 1)
    np.add.at(A[:, 7], pc[det], (pred[det] == lab_idx[det]).astype(float))
    return A


def _stats(v):
    n, c = v[:3], v[3:6]
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(n > 0, c / n, np.nan)
    acc = c.sum() / max(n.sum(), 1)
    maj = n.max() / max(n.sum(), 1)
    return acc, float(np.nanmean(rec)), maj, acc - maj


def rescore(preds, labels, n_boot=2000, seed=0):
    t = labels.merge(preds, on=["recording", "event_idx"], how="left", suffixes=("", "_p"))
    if t["processed"].isna().any():
        print(f"ATTENTION : {int(t['processed'].isna().sum())} événements du tableau de labels sans prédiction (ignorés)")
    t = t[t["processed"] == True]
    rng = np.random.default_rng(seed)
    variants = [c for c in labels.columns if c not in ("recording", "event_idx", "patient")]
    out = []
    for name in variants + ["événements focaux seulement"]:
        if name == "événements focaux seulement":
            sub = t[t["focal"] == True]
            lab = sub["référence"].values
        else:
            sub = t
            lab = sub[name].values
        ok = lab != "unlabeled"
        sub, lab = sub[ok], lab[ok]
        li = np.array([CIDX[x] for x in lab])
        A = _counts(li, sub["pred"].values, sub["patient"].values)
        v = A.sum(0)
        acc, bacc, maj, gap = _stats(v)
        idx = rng.integers(0, len(A), (n_boot, len(A)))
        B = A[idx].sum(1)
        bs = np.array([_stats(b) for b in B])
        lo = np.nanpercentile(bs, 2.5, axis=0)
        hi = np.nanpercentile(bs, 97.5, axis=0)
        out.append(dict(variante=name, n_evenements=int(v[:3].sum()), G=int(v[0]), D=int(v[1]), B=int(v[2]),
                        exactitude=acc, exactitude_ic=(lo[0], hi[0]), equilibree=bacc, equilibree_ic=(lo[1], hi[1]),
                        majoritaire=maj, ecart=gap, ecart_ic=(lo[3], hi[3]),
                        exactitude_parmi_detectes=v[7] / max(v[6], 1)))
    return pd.DataFrame(out)


def rescore_report(res):
    L = ["variante | n | G/D/B | exactitude [IC95 par patient] | équilibrée [IC95] | classe majoritaire | exactitude - majoritaire [IC95]"]
    for r in res.itertuples():
        L.append(f"{r.variante} | {r.n_evenements} | {r.G}/{r.D}/{r.B} | {r.exactitude:.3f} [{r.exactitude_ic[0]:.3f}, {r.exactitude_ic[1]:.3f}] | "
                 f"{r.equilibree:.3f} [{r.equilibree_ic[0]:.3f}, {r.equilibree_ic[1]:.3f}] | {r.majoritaire:.3f} | {100 * r.ecart:+.1f} pts [{100 * r.ecart_ic[0]:+.1f}, {100 * r.ecart_ic[1]:+.1f}]")
    ref = res[res.variante == "référence"].iloc[0]
    L.append(f"CONTRÔLE : la ligne « référence » doit donner exactitude 0.398 et équilibrée 0.435 (rapport 5b) ; ici {ref.exactitude:.3f} et {ref.equilibree:.3f} sur {ref.n_evenements} événements")
    sp = res.equilibree.drop(res.index[res.variante == "événements focaux seulement"]).agg(["min", "max"])
    L.append(f"exactitude équilibrée selon les variantes de label : de {sp['min']:.3f} à {sp['max']:.3f} ; "
             f"écart à la classe majoritaire : de {100 * res.ecart.min():+.1f} à {100 * res.ecart.max():+.1f} points")
    return L


# ----------------------------------------------------------------------------------------------------------------------
def main_confusion(a):
    L, _ = confusion_report(a.scores)
    txt = "\n".join(["<<<COPY_FROM_HERE>>>"] + L + ["<<<COPY_TO_HERE>>>"])
    print(txt)
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        open(os.path.join(a.outdir, "rapport_confusion.txt"), "w", encoding="utf-8").write(txt + "\n")


def main_rescore(a):
    preds, P, m2key = build_predictions(a.manifest, a.params, a.m2key)
    labels = pd.read_csv(a.labels)
    res = rescore(preds, labels, a.n_boot)
    os.makedirs(a.outdir, exist_ok=True)
    preds.to_csv(os.path.join(a.outdir, "predictions_figees_eval.csv"), index=False)
    res.to_csv(os.path.join(a.outdir, "re_notation_eval.csv"), index=False)
    L = [f"règle figée : tau={P['tau']}, n_min={P['n_min']}, fusion {P.get('merge_gap_s', 6.0)} s, M2 = {m2key} ; "
         f"{int(preds.processed.sum())} événements dans les enregistrements traités, {int((~preds.processed).sum())} dans les rejetés (non comptés)"] + rescore_report(res)
    txt = "\n".join(["<<<COPY_FROM_HERE>>>"] + L + ["<<<COPY_TO_HERE>>>"])
    print(txt)
    open(os.path.join(a.outdir, "rapport_re_notation.txt"), "w", encoding="utf-8").write(txt + "\n")


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    # matrice de confusion
    tmp = tempfile.mkdtemp(prefix="step27_selftest_")
    try:
        rows = []
        for p in range(6):
            for e in range(4):
                y = (p + e) % 3
                probs = np.full(3, 0.1)
                probs[y if e != 3 else (y + 1) % 3] = 0.8
                for w in range(2):
                    rows.append(dict(recording=f"r{p}", eid=e, patient=f"p{p}", y=y, pL=probs[0], pR=probs[1], pB=probs[2]))
        csv = os.path.join(tmp, "s.csv")
        pd.DataFrame(rows).to_csv(csv, index=False)
        L, C = confusion_report(csv)
        check("confusion : 24 événements comptés", C.sum() == 24)
        check("confusion : 6 événements faux (1 par patient)", C.sum() - np.trace(C) == 6)

        # prédiction d'événements : comparaison avec step5b.eval_dets
        from step5b_pipeline_eval import eval_dets, detect, FS
        starts = np.arange(0, 60 * 500, 500)                          # 60 fenêtres de 4 s, pas de 2 s
        p1 = np.full(len(starts), 0.05)
        p2 = np.tile([0.1, 0.1, 0.8], (len(starts), 1)).astype(float)
        p1[5:9] = 0.95                                                 # détection autour de 10-20 s
        p2[5:9] = [0.8, 0.1, 0.1]                                      # côté gauche
        p1[30:33] = 0.95                                               # détection autour de 60-68 s
        p2[30:33] = [0.1, 0.8, 0.1]                                    # côté droit
        sc = dict(starts=starts, p1=p1, flat=np.zeros(len(starts), bool), n_samples=len(starts) * 500 + 500, p2_x=p2)
        S = [(11.0, 19.0), (61.0, 66.0), (100.0, 110.0)]               # la troisième n'est pas détectée
        spans = [(11.0, 19.0), (61.0, 66.0), (100.0, 110.0)]
        pr = predict_events(sc, S, spans, 0.8, 1, "x")
        check(f"prédictions par événement : gauche, droite, manqué ({pr})", pr == [0, 1, -1])
        v = eval_dets(sc, S, [(a, b, y) for (a, b), y in zip(spans, [0, 1, 2])], detect(sc["starts"], sc["p1"], sc["flat"], 0.8, 1, 6.0), "x")
        check("cohérence avec step5b.eval_dets (corrects par classe 1/1/0)", list(v[8:11]) == [1, 1, 0])

        # re-notation
        preds = pd.DataFrame(dict(recording=["a", "a", "b", "b", "c"], event_idx=[0, 1, 0, 1, 0], patient=["p1", "p1", "p2", "p2", "p3"],
                                  processed=[True] * 5, pred=[0, 2, 1, -1, 2], focal=[True, True, False, True, False]))
        labels = pd.DataFrame(dict(recording=["a", "a", "b", "b", "c"], event_idx=[0, 1, 0, 1, 0], patient=["p1", "p1", "p2", "p2", "p3"],
                                   **{"référence": ["left", "bilateral", "right", "left", "bilateral"],
                                      "k = 1": ["left", "bilateral", "right", "bilateral", "bilateral"]}))
        res = rescore(preds, labels, n_boot=200)
        r0 = res[res.variante == "référence"].iloc[0]
        check(f"référence : 4 événements corrects sur 5, exactitude {r0.exactitude:.2f}", abs(r0.exactitude - 0.8) < 1e-9 and r0.n_evenements == 5)
        check("référence : classe majoritaire 2/5", abs(r0.majoritaire - 0.4) < 1e-9)
        r1 = res[res.variante == "k = 1"].iloc[0]
        check(f"variante k = 1 : 3 corrects sur 5 (exactitude {r1.exactitude:.2f}) -- le label change pour l'événement manqué", abs(r1.exactitude - 0.8) < 1e-9 or abs(r1.exactitude - 0.6) < 1e-9)
        f = res[res.variante == "événements focaux seulement"].iloc[0]
        check("focal seulement : 3 événements", f.n_evenements == 3)
        check("rapport écrit", len(rescore_report(res)) >= 4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    base = "/content/drive/MyDrive/projet_propre/"
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["confusion", "rescore"])
    ap.add_argument("--scores", default=base + "lr_cv/B_asym80_val_scores.csv")
    ap.add_argument("--manifest", default=base + "scores/scores_manifest.csv")
    ap.add_argument("--params", default=base + "scores/params_v2.json")
    ap.add_argument("--m2key", default=None)
    ap.add_argument("--labels", default=base + "label_sensitivity/labels_variantes_eval.csv")
    ap.add_argument("--outdir", default=base + "label_sensitivity")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.cmd:
        ap.error("donner confusion ou rescore (ou --selftest)")
    {"confusion": main_confusion, "rescore": main_rescore}[a.cmd](a)
