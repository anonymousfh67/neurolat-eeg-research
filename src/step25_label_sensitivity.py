#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 25 -- Sensibilité des labels gauche / droite / bilatéral à la règle qui les définit (annotations seules, CPU, quelques minutes).

C'est le « script 5 » d'origine (tableau 1 de l'article), porté vers la règle ACTUELLE de l'étape 1 :
  * comptage en CANAUX DISTINCTS (le script 5 ancien comptait des lignes d'annotation),
  * égalité -> bilatéral (l'ancien script : égalité -> droite),
  * TOUS les événements de chaque enregistrement (l'ancien script : le premier seulement, avec un filtre 0,5-15 Mo sur l'EDF).
Aucun signal EEG n'est lu, aucun modèle n'est entraîné ; l'analyse peut donc porter sur train, dev et eval sans aucune conséquence sur l'évaluation.

Pour chaque variante : nombre d'événements dont le label change, pourcentage avec intervalle bootstrap par patient, kappa de Cohen contre la référence,
répartition G/D/B, taux de la classe majoritaire, nombre de changements qui touchent « bilatéral », nombre de passages directs gauche <-> droite.
La ligne « ancienne règle de l'article » (comptage en lignes + égalité -> droite) mesure combien de labels diffèrent entre l'ancienne et la nouvelle règle.
« Événements focaux seulement » retire des événements au lieu de les ré-étiqueter : pas de pourcentage de changement.

Usage :
  python step25_label_sensitivity.py [--root .../edf] [--splits eval dev train] [--outdir ...]
  python step25_label_sensitivity.py --selftest
"""
import argparse, os, sys, tempfile, shutil
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_inventaire_labels import parse_annotation, extract_events, onset_counts, label_of, REF

VARIANTS = [
    ("référence (W = 2 s, k = 2)", {}),
    ("fenêtre d'onset W = 1 s", dict(W=1.0)),
    ("fenêtre d'onset W = 3 s", dict(W=3.0)),
    ("fenêtre d'onset W = 5 s", dict(W=5.0)),
    ("égalité -> droite", dict(tie="right")),
    ("k = 1 canal par côté", dict(k=1)),
    ("k = 3 canaux par côté", dict(k=3)),
    ("comptage en lignes (pas en canaux)", dict(count="rows")),
    ("ancienne règle de l'article (lignes + égalité -> droite)", dict(tie="right", count="rows")),
]
CLASSES = ("left", "right", "bilateral")


def load_events(root, split):
    """Tous les événements (fnsz / cpsz / gnsz fusionnés à 1 s) des annotations .csv de la partition. Aucun accès aux EDF."""
    out = []
    d = os.path.join(root, split)
    if not os.path.isdir(d):
        raise SystemExit(f"Dossier introuvable : {d}")
    for dirpath, dirs, files in os.walk(d):
        dirs.sort()
        for fn in sorted(files):
            if not fn.endswith(".csv"):
                continue
            stem = fn[:-4]
            _, rows = parse_annotation(os.path.join(dirpath, fn))
            for i, ev in enumerate(extract_events(rows)):
                ev.update(recording=stem, patient=stem.split("_")[0], event_idx=i)
                out.append(ev)
    return out


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in CLASSES)
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def boot_pct(changed, patient, B=2000, seed=0):
    """IC 95 % bootstrap PAR PATIENT du pourcentage d'événements qui changent."""
    pc, _ = pd.factorize(pd.Series(patient))
    P = int(pc.max()) + 1
    sc = np.bincount(pc, weights=np.asarray(changed, float), minlength=P)
    sn = np.bincount(pc, minlength=P).astype(float)
    idx = np.random.default_rng(seed).integers(0, P, size=(B, P))
    r = 100 * sc[idx].sum(1) / sn[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def analyse(events, n_boot=2000):
    """Retourne (tableau par variante, tableau d'événements). Seuls les événements étiquetés par la référence sont comparés."""
    ref = [label_of(e, REF) for e in events]
    keep = [i for i, l in enumerate(ref) if l is not None]
    ev = [events[i] for i in keep]
    r = np.array([ref[i] for i in keep])
    patient = np.array([e["patient"] for e in ev])
    rows, table = [], pd.DataFrame(dict(recording=[e["recording"] for e in ev], event_idx=[e["event_idx"] for e in ev], patient=patient, **{"référence": r}))
    for name, over in VARIANTS:
        cfg = {**REF, **over}
        lab = np.array([label_of(e, cfg) or "unlabeled" for e in ev])
        table[name] = lab
        chg = lab != r
        both = lab != "unlabeled"
        lo, hi = boot_pct(chg, patient, n_boot) if chg.any() or len(r) else (float("nan"),) * 2
        c = Counter(lab)
        n_lab = int(both.sum())
        rows.append(dict(variante=name, n_evenements=len(r), n_changes=int(chg.sum()), pct=100 * chg.mean(), ic_bas=lo, ic_haut=hi,
                         kappa=kappa(lab[both], r[both]), G=c["left"], D=c["right"], B=c["bilateral"], sans_label=c["unlabeled"],
                         majoritaire_pct=100 * max(c["left"], c["right"], c["bilateral"]) / max(n_lab, 1),
                         changes_avec_B=int((chg & ((lab == "bilateral") | (r == "bilateral"))).sum()),
                         passages_G_D=int((chg & (((r == "left") & (lab == "right")) | ((r == "right") & (lab == "left")))).sum())))
    foc = [e for e in ev if onset_counts(e, REF["W"], REF["count"])[0]]
    lf = Counter(label_of(e, REF) for e in foc)
    rows.append(dict(variante=f"événements focaux seulement ({len(foc)} événements)", n_evenements=len(foc), n_changes=np.nan, pct=np.nan, ic_bas=np.nan,
                     ic_haut=np.nan, kappa=np.nan, G=lf["left"], D=lf["right"], B=lf["bilateral"], sans_label=0,
                     majoritaire_pct=100 * max(lf["left"], lf["right"], lf["bilateral"]) / max(len(foc), 1), changes_avec_B=np.nan, passages_G_D=np.nan))
    return pd.DataFrame(rows), table


def report(split, events, sens, table):
    n, npat = len(table), table.patient.nunique()
    L = [f"=== Partition {split} : {n} événements étiquetés par la référence, {npat} patients "
         f"({sum(1 for e in events if label_of(e, REF) is None)} événements sans canal gauche/droit au début, non comptés) ===",
         "variante | n changés | % [IC 95 % par patient] | kappa | G/D/B | classe majoritaire | changements touchant B | passages directs G<->D"]
    for r in sens.itertuples():
        if np.isnan(r.pct):
            L.append(f"{r.variante} | - | - | - | {r.G}/{r.D}/{r.B} | {r.majoritaire_pct:.1f} % | - | -")
        else:
            L.append(f"{r.variante} | {int(r.n_changes)} | {r.pct:.1f} % [{r.ic_bas:.1f} ; {r.ic_haut:.1f}] | {r.kappa:.2f} | {r.G}/{r.D}/{r.B}"
                     f"{'+' + str(r.sans_label) + ' sans label' if r.sans_label else ''} | {r.majoritaire_pct:.1f} % | {int(r.changes_avec_B)} | {int(r.passages_G_D)}")
    pm = table.groupby("patient")["référence"].nunique()
    L.append(f"patients ayant des labels différents : {int((pm > 1).sum())}/{len(pm)} ; événements par patient : médiane {table.groupby('patient').size().median():g}")
    return L


def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    L = ["<<<COPY_FROM_HERE>>>", f"Règle de référence (étape 1) : {REF}"]
    for split in a.splits:
        events = load_events(a.root, split)
        sens, table = analyse(events, a.n_boot)
        sens.to_csv(os.path.join(a.outdir, f"sensibilite_{split}.csv"), index=False)
        table.to_csv(os.path.join(a.outdir, f"labels_variantes_{split}.csv"), index=False)
        L += report(split, events, sens, table) + [""]
    L.append("<<<COPY_TO_HERE>>>")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(a.outdir, "rapport_sensibilite.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


def selftest():
    from step1_inventaire_labels import _write
    tmp = tempfile.mkdtemp(prefix="step25_selftest_")
    L2, R2 = ["FP1-F7", "F7-T3"], ["FP2-F8", "F8-T4"]
    cases = {
        "pa000001_s001_t000": [(c, 10, 20, "fnsz") for c in L2],                                                    # gauche partout
        "pb000002_s001_t000": [(c, 10, 20, "fnsz") for c in L2 + R2],                                               # bilatéral
        "pc000003_s001_t000": [("FP1-F7", 10, 20, "fnsz"), ("FP2-F8", 10, 20, "fnsz")],                             # égalité 1-1 : B ; droite si égalité -> droite ; B si k=1
        "pd000004_s001_t000": [(c, 10, 30, "fnsz") for c in L2] + [(c, 14, 30, "fnsz") for c in R2],                # G à W=2 ; B à W=5
        "pe000005_s001_t000": [(c, 10, 20, "gnsz") for c in ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP1-F3"]] + [("FP2-F8", 10, 20, "gnsz")],   # G ; B si k=1
        "pf000006_s001_t000": [("FP1-F7", 10, 20, "fnsz")] * 3 + [("FP2-F8", 10, 20, "fnsz")],                      # B (1 contre 1 canal) ; G en lignes (3 contre 1)
        "pg000007_s001_t000": [("FZ-CZ", 10, 20, "fnsz")],                                                          # aucun canal G/D : non compté
    }
    for stem, rows in cases.items():
        _write(tmp, "eval", stem, rows)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    ev = load_events(tmp, "eval")
    check(f"7 événements lus ({len(ev)})", len(ev) == 7)
    sens, table = analyse(ev, n_boot=200)
    got = dict(zip(table.recording.str[:2], table["référence"]))
    check(f"labels de référence {got}", got == {"pa": "left", "pb": "bilateral", "pc": "bilateral", "pd": "left", "pe": "left", "pf": "bilateral"})
    check("l'événement hors montage n'est pas compté (6 événements comparés)", len(table) == 6)
    s = sens.set_index("variante")
    exp = {"référence (W = 2 s, k = 2)": 0, "fenêtre d'onset W = 1 s": 0, "fenêtre d'onset W = 5 s": 1, "égalité -> droite": 2, "k = 1 canal par côté": 1,
           "k = 3 canaux par côté": 0, "comptage en lignes (pas en canaux)": 1, "ancienne règle de l'article (lignes + égalité -> droite)": 2}
    for k_, v in exp.items():
        check(f"{k_} : {v} changement(s) (obtenu {int(s.loc[k_, 'n_changes'])})", int(s.loc[k_, "n_changes"]) == v)
    check("aucun passage direct gauche <-> droite dans ces cas", int(s["passages_G_D"].iloc[:-1].sum()) == 0)
    check("la ligne « égalité -> droite » compte 2 changements touchant B", int(s.loc["égalité -> droite", "changes_avec_B"]) == 2)
    check("kappa = 1 quand rien ne change", abs(s.loc["référence (W = 2 s, k = 2)", "kappa"] - 1) < 1e-9)
    check("kappa < 1 quand quelque chose change", s.loc["égalité -> droite", "kappa"] < 1)
    check("ligne « focaux seulement » : 5 événements focaux sur 6", int(s.iloc[-1]["n_evenements"]) == 5)
    lo, hi = s.loc["égalité -> droite", "ic_bas"], s.loc["égalité -> droite", "ic_haut"]
    check(f"IC bootstrap encadre le pourcentage ({lo:.1f} <= {s.loc['égalité -> droite', 'pct']:.1f} <= {hi:.1f})", lo <= s.loc["égalité -> droite", "pct"] <= hi)
    out = os.path.join(tmp, "out")
    main(argparse.Namespace(root=tmp, splits=["eval"], outdir=out, n_boot=100))
    check("rapport écrit", os.path.exists(os.path.join(out, "rapport_sensibilite.txt")))
    shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("TUSZ_ROOT", "/content/drive/MyDrive/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf"))
    ap.add_argument("--splits", nargs="+", default=["eval", "dev", "train"])
    ap.add_argument("--outdir", default="/content/drive/MyDrive/projet_propre/label_sensitivity")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest()) if a.selftest else main(a)
