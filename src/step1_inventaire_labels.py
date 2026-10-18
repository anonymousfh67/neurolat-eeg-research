#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 1 -- Inventaire des données et définition des labels (annotations .csv UNIQUEMENT).
Aucun signal EEG n'est lu, aucun modèle n'est entraîné, aucun GPU n'est nécessaire (1 à 5 minutes).

Ce que fait ce script
  1. Parcourt TUSZ train / dev / eval (fichiers .csv d'annotations).
  2. Vérifie qu'AUCUN patient n'apparaît dans deux partitions.
  3. Extrait TOUS les événements de crise (fnsz, cpsz, gnsz ; fusion si écart <= 1 s).
  4. Attribue à chaque événement un label gauche / droite / bilatéral par UNE règle explicite.
  5. Mesure la sensibilité des labels aux paramètres de la règle (W, k, égalités, ratio, comptage).

Règle de référence (identique dans le code et dans la description écrite)
  - Fenêtre d'onset : lignes d'annotation qui démarrent au plus tard à t0 + W   (W = 2 s)
  - nL, nR = nombre de CANAUX DISTINCTS annotés à gauche / à droite dans cette fenêtre
  - Événement avec annotation focale (fnsz ou cpsz) : seules les lignes focales sont comptées
        bilatéral si nL >= k et nR >= k ; sinon le côté le plus fréquent ; égalité -> bilatéral
  - Événement généralisé seul (gnsz) :
        bilatéral si nL >= k et nR >= k ; gauche si nL > rho*nR ; droite si nR > rho*nL ; sinon bilatéral
  - Si nL = nR = 0 : l'événement n'est PAS labélisable (statut 'unlabeled'), il n'est pas deviné.
  (k = 2, rho = 2.)

Usage
    python step1_inventaire_labels.py --root /chemin/tuh_eeg_seizure/v2.0.3/edf --outdir sortie_etape1
    python step1_inventaire_labels.py --selftest
Dépendances : numpy, pandas.
"""
import argparse, os, re, sys, tempfile, shutil, time
from collections import Counter
from itertools import product, combinations
import numpy as np
import pandas as pd

SPLITS = ("train", "dev", "eval")

# Montage d'annotation TUSZ (TCP, 22 canaux) : 11 canaux gauches et 11 droits.
LEFT = {"FP1-F7", "F7-T3", "T3-T5", "T5-O1", "A1-T3", "T3-C3", "C3-CZ", "FP1-F3", "F3-C3", "C3-P3", "P3-O1"}
RIGHT = {"FP2-F8", "F8-T4", "T4-T6", "T6-O2", "CZ-C4", "C4-T4", "T4-A2", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"}

LATERALIZABLE = ("fnsz", "cpsz", "gnsz")   # types de crise dont on dérive un label
FOCAL = ("fnsz", "cpsz")
MERGE_GAP = 1.0                              # secondes
REF = dict(W=2.0, k=2, rho=2.0, tie="bilateral", count="channels")


# ----------------------------------------------------------------------------- lecture
def parse_annotation(path):
    """Retourne (durée en s ou None, liste de (canal, début, fin, label))."""
    duration, rows = None, []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                m = re.search(r"duration\s*=\s*([0-9.]+)", s)
                if m:
                    duration = float(m.group(1))
                continue
            p = [x.strip() for x in s.split(",")]
            if len(p) < 4 or p[0].lower() == "channel":
                continue
            try:
                a, b = float(p[1]), float(p[2])
            except ValueError:
                continue
            rows.append((p[0].upper(), a, b, p[3].lower()))
    return duration, rows


def extract_events(rows, gap=MERGE_GAP):
    """Fusionne les lignes fnsz/cpsz/gnsz dont l'écart est <= gap en événements."""
    sz = sorted([r for r in rows if r[3] in LATERALIZABLE], key=lambda r: r[1])
    events, cur = [], None
    for r in sz:
        if cur is None:
            cur = dict(start=r[1], stop=r[2], rows=[r])
        elif r[1] <= cur["stop"] + gap:
            cur["stop"] = max(cur["stop"], r[2])
            cur["rows"].append(r)
        else:
            events.append(cur)
            cur = dict(start=r[1], stop=r[2], rows=[r])
    if cur:
        events.append(cur)
    return events


# ----------------------------------------------------------------------------- règle
def onset_counts(ev, W, count):
    has_focal = any(r[3] in FOCAL for r in ev["rows"])
    use = [r for r in ev["rows"] if r[3] in FOCAL] if has_focal else ev["rows"]
    onset = [r for r in use if r[1] <= ev["start"] + W]
    L = [r[0] for r in onset if r[0] in LEFT]
    R = [r[0] for r in onset if r[0] in RIGHT]
    if count == "channels":
        return has_focal, len(set(L)), len(set(R))
    return has_focal, len(L), len(R)


def apply_rule(has_focal, nL, nR, k, rho, tie):
    if nL == 0 and nR == 0:
        return None
    if nL >= k and nR >= k:
        return "bilateral"
    if has_focal:
        if nL > nR:
            return "left"
        if nR > nL:
            return "right"
        return tie
    if nL > rho * nR:
        return "left"
    if nR > rho * nL:
        return "right"
    return "bilateral"


def label_of(ev, cfg):
    has_focal, nL, nR = onset_counts(ev, cfg["W"], cfg["count"])
    return apply_rule(has_focal, nL, nR, cfg["k"], cfg["rho"], cfg["tie"])


# ----------------------------------------------------------------------------- inventaire
def run(root, outdir, splits=SPLITS, quiet=False, stat_edf=True):
    os.makedirs(outdir, exist_ok=True)
    t_start = time.time()
    rec_rows, ev_rows, ev_objs = [], [], []
    for split in splits:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            raise SystemExit(f"Dossier introuvable : {d}")
        n_seen = 0
        for dirpath, dirs, files in os.walk(d):
            dirs.sort()
            for fn in sorted(files):
                if not fn.endswith(".csv"):
                    continue
                n_seen += 1
                if not quiet and n_seen % 200 == 0:
                    print(f"  [{split}] {n_seen} fichiers .csv lus ({time.time() - t_start:.0f} s)", flush=True)
                path = os.path.join(dirpath, fn)
                stem = fn[:-4]
                parts = stem.split("_")
                patient, session = parts[0], (parts[1] if len(parts) > 1 else "")
                duration, rows = parse_annotation(path)
                edf = path[:-4] + ".edf"
                if stat_edf:
                    edf_ok = os.path.exists(edf)
                    edf_mb = round(os.path.getsize(edf) / 1024 ** 2, 3) if edf_ok else None
                else:
                    edf_ok, edf_mb = None, None   # non vérifié (plus rapide sur Google Drive)
                events = extract_events(rows)
                labels = []
                for i, ev in enumerate(events):
                    hf, nLc, nRc = onset_counts(ev, REF["W"], "channels")
                    _, nLr, nRr = onset_counts(ev, REF["W"], "rows")
                    lab = label_of(ev, REF)
                    lab_rows = label_of(ev, {**REF, "count": "rows"})
                    labels.append(lab)
                    ev_rows.append(dict(
                        split=split, recording=stem, patient=patient, session=session, event_idx=i,
                        start=ev["start"], stop=ev["stop"], has_focal=hf,
                        types="+".join(sorted({r[3] for r in ev["rows"]})),
                        nL_channels=nLc, nR_channels=nRc, nL_rows=nLr, nR_rows=nRr,
                        label=lab or "unlabeled", label_if_counting_rows=lab_rows or "unlabeled"))
                    ev_objs.append((split, ev))
                rec_rows.append(dict(
                    split=split, recording=stem, patient=patient, session=session,
                    edf_exists=edf_ok, edf_mb=edf_mb,
                    duration_s=duration, n_rows=len(rows),
                    n_seizure_rows_any_type=sum(1 for r in rows if r[3] != "bckg"),
                    n_events=len(events), n_labeled_events=sum(l is not None for l in labels)))
    rec = pd.DataFrame(rec_rows)
    ev = pd.DataFrame(ev_rows)

    # --- vérification : aucun patient dans deux partitions
    pats = {s: set(rec.loc[rec.split == s, "patient"]) for s in splits}
    for a, b in combinations(splits, 2):
        inter = pats[a] & pats[b]
        if inter:
            raise SystemExit(f"ERREUR : {len(inter)} patient(s) présents dans {a} ET {b} : {sorted(inter)[:10]}")

    # --- sensibilité
    sens_rows = []
    for split in splits:
        objs = [e for s, e in ev_objs if s == split]
        if not objs:
            continue
        base = [label_of(e, REF) for e in objs]
        cfgs = [(f"W={W:g}s, k={k}", {**REF, "W": W, "k": k}) for W, k in product((1.0, 2.0, 3.0, 5.0), (1, 2, 3))]
        cfgs += [("égalité -> droite", {**REF, "tie": "right"}), ("égalité -> gauche", {**REF, "tie": "left"}),
                 ("rho = 1,5", {**REF, "rho": 1.5}), ("rho = 3", {**REF, "rho": 3.0}),
                 ("comptage en lignes (pas en canaux)", {**REF, "count": "rows"})]
        for name, cfg in cfgs:
            lab = [label_of(e, cfg) for e in objs]
            c = Counter(l or "unlabeled" for l in lab)
            labeled = [l for l in lab if l]
            sens_rows.append(dict(
                split=split, configuration=name, n_events=len(objs),
                n_changed=sum(1 for a, b in zip(base, lab) if a != b),
                pct_changed=round(100 * sum(1 for a, b in zip(base, lab) if a != b) / len(objs), 1),
                left=c["left"], right=c["right"], bilateral=c["bilateral"], unlabeled=c["unlabeled"],
                majority_rate_pct=round(100 * max(c["left"], c["right"], c["bilateral"]) / max(len(labeled), 1), 1)))
    sens = pd.DataFrame(sens_rows)

    rec.to_csv(os.path.join(outdir, "recordings.csv"), index=False)
    ev.to_csv(os.path.join(outdir, "events.csv"), index=False)
    sens.to_csv(os.path.join(outdir, "sensitivity.csv"), index=False)

    # --- rapport
    L = ["<<<COPY_FROM_HERE>>>", f"Règle de référence : {REF}",
         f"Patients communs entre les partitions lues ({', '.join(splits)}) : aucun (vérifié)"]
    for split in splits:
        r, e = rec[rec.split == split], ev[ev.split == split]
        lab = e[e.label != "unlabeled"]
        c = Counter(lab.label)
        L.append(f"\n--- {split} ---")
        edf_txt = str(int(r.edf_exists.sum())) if r.edf_exists.notna().any() else "non vérifié"
        L.append(f"enregistrements (.csv) : {len(r)} | avec .edf : {edf_txt} | patients : {r.patient.nunique()}")
        L.append(f"enregistrements avec >= 1 événement fnsz/cpsz/gnsz : {int((r.n_events > 0).sum())} "
                 f"| patients concernés : {e.patient.nunique()}")
        L.append(f"événements : {len(e)} | labélisés : {len(lab)} | non labélisables (aucun canal gauche/droit à l'onset) : {len(e) - len(lab)}")
        if len(lab):
            L.append(f"gauche / droite / bilatéral : {c['left']} / {c['right']} / {c['bilateral']} "
                     f"| taux de la classe majoritaire : {100 * max(c.values()) / len(lab):.1f} %")
            per = lab.groupby("patient").size()
            mixed = int((lab.groupby("patient").label.nunique() > 1).sum())
            L.append(f"événements labélisés par patient : médiane {per.median():g}, min {per.min()}, max {per.max()} "
                     f"| patients ayant des labels différents : {mixed}/{len(per)}")
            L.append(f"événements sans annotation focale (généralisés seuls) : {int((~lab.has_focal).sum())}")
            diff = int((lab.label != lab.label_if_counting_rows).sum())
            L.append(f"labels qui changeraient si on comptait des lignes au lieu de canaux distincts : {diff}")
    show = "eval" if "eval" in splits else splits[0]
    L.append(f"\nSensibilité des labels (partition {show}) : % d'événements dont le label change par rapport à la référence")
    s = sens[sens.split == show]
    if len(s):
        L.append(s.drop(columns="split").to_string(index=False))
    L.append("<<<COPY_TO_HERE>>>")
    txt = "\n".join(L)
    with open(os.path.join(outdir, "inventory_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    if not quiet:
        print(txt)
        print(f"\nFichiers écrits dans {outdir} : recordings.csv, events.csv, sensitivity.csv, inventory_report.txt")
    return rec, ev, sens


# ----------------------------------------------------------------------------- auto-test
HEADER = "# version = csv_v1.0.0\n# bname = {b}\n# duration = 300.00 secs\n#\nchannel,start_time,stop_time,label,confidence\n"


def _write(root, split, stem, rows):
    d = os.path.join(root, split, stem.split("_")[0], "s001", "01_tcp_ar")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, stem + ".csv"), "w") as f:
        f.write(HEADER.format(b=stem))
        f.write("FP1-F7,0.0000,5.0000,bckg,1.0000\n")
        for ch, a, b, lab in rows:
            f.write(f"{ch},{a:.4f},{b:.4f},{lab},1.0000\n")
    open(os.path.join(d, stem + ".edf"), "wb").write(b"0" * 1024)


def selftest():
    tmp = tempfile.mkdtemp(prefix="step1_selftest_")
    L2, R2 = ["FP1-F7", "F7-T3"], ["FP2-F8", "F8-T4"]
    cases = {  # nom : (lignes, label attendu avec la règle de référence)
        "pa000001_s001_t000": ([(c, 10, 20, "fnsz") for c in L2], "left"),
        "pa000002_s001_t000": ([(c, 10, 20, "fnsz") for c in L2 + R2], "bilateral"),
        "pa000003_s001_t000": ([("FP1-F7", 10, 20, "fnsz"), ("FP2-F8", 10, 20, "fnsz")], "bilateral"),   # égalité 1-1
        "pa000004_s001_t000": ([(c, 10, 20, "gnsz") for c in ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP1-F3"]]
                               + [("FP2-F8", 10, 20, "gnsz")], "left"),                                   # 5 > 2*1
        "pa000005_s001_t000": ([(c, 10, 20, "gnsz") for c in L2 + R2], "bilateral"),
        "pa000006_s001_t000": ([("FP1-F7", 10, 20, "fnsz")] * 3 + [("FP2-F8", 10, 20, "fnsz")], "bilateral"),
        # comptage en lignes : nL=3, nR=1 -> gauche ; en canaux distincts : 1 contre 1 -> égalité -> bilatéral
        "pa000007_s001_t000": ([(c, 10, 20, "fnsz") for c in L2] + [(c, 20.5, 30, "fnsz") for c in L2]
                               + [(c, 100, 110, "fnsz") for c in R2], "left"),                            # 2 événements
        "pa000008_s001_t000": ([(c, 10, 30, "fnsz") for c in L2] + [(c, 14, 30, "fnsz") for c in R2], "left"),
        "pa000009_s001_t000": ([("FZ-CZ", 10, 20, "fnsz")], "unlabeled"),                                 # hors montage gauche/droit
        "pa000010_s001_t000": ([("FP1-F7", 10, 20, "tcsz")], None),                                       # type exclu : aucun événement
        "pa000011_s001_t000": ([(c, 10, 20, "gnsz") for c in R2] + [(c, 10, 20, "fnsz") for c in L2], "left"),  # focal prioritaire
    }
    for stem, (rows, _) in cases.items():
        _write(tmp, "train", stem, rows)
    _write(tmp, "dev", "pb000001_s001_t000", [(c, 10, 20, "fnsz") for c in L2])
    _write(tmp, "eval", "pc000001_s001_t000", [(c, 10, 20, "fnsz") for c in R2])
    out = os.path.join(tmp, "out")
    rec, ev, sens = run(tmp, out, quiet=True)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    tr = ev[ev.split == "train"]
    for stem, (_, exp) in cases.items():
        got = tr[(tr.recording == stem) & (tr.event_idx == 0)].label
        if exp is None:
            check(f"{stem} : aucun événement", got.empty)
        else:
            check(f"{stem} : attendu {exp}", (not got.empty) and got.iloc[0] == exp)
    check("pa000006 : en comptant des lignes -> gauche",
          tr[tr.recording == "pa000006_s001_t000"].label_if_counting_rows.iloc[0] == "left")
    check("pa000007 : fusion (écart 0,5 s) + 2e événement séparé = 2 événements",
          int((tr.recording == "pa000007_s001_t000").sum()) == 2)
    o = [s for s in sens.itertuples() if s.split == "train" and s.configuration == "W=5s, k=2"][0]
    check("W = 5 s change le label de pa000008 (gauche -> bilatéral) : au moins 1 changement", o.n_changed >= 1)
    # égalité -> droite : change les 2 événements à égalité de canaux distincts (pa000003 et pa000006)
    o = [s for s in sens.itertuples() if s.split == "train" and s.configuration == "égalité -> droite"][0]
    check("égalité -> droite change exactement 2 événements (pa000003, pa000006)", o.n_changed == 2)
    chg = tr[tr.recording.isin(["pa000003_s001_t000", "pa000006_s001_t000"])]
    check("ces 2 événements sont bien des égalités 1 contre 1 en canaux distincts",
          bool(((chg.nL_channels == 1) & (chg.nR_channels == 1)).all()))
    rec2, ev2, _ = run(tmp, out + "_fast", quiet=True, stat_edf=False)
    check("mode --no_edf_stat : mêmes labels, edf non vérifié",
          list(ev2.label) == list(ev.label) and rec2.edf_exists.isna().all())
    # détection d'un patient commun à deux partitions
    _write(tmp, "eval", "pb000001_s002_t000", [(c, 10, 20, "fnsz") for c in R2])
    try:
        run(tmp, out + "2", quiet=True)
        check("patient commun dev/eval détecté", False)
    except SystemExit:
        check("patient commun dev/eval détecté (arrêt avec erreur)", True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("TUSZ_ROOT", "/content/drive/MyDrive/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf"),
                    help="dossier contenant train/ dev/ eval/")
    ap.add_argument("--outdir", default="sortie_etape1")
    ap.add_argument("--splits", default="train,dev,eval", help="ex. eval  (commencer par la plus petite partition)")
    ap.add_argument("--no_edf_stat", action="store_true", help="ne pas interroger les .edf (beaucoup plus rapide sur Drive)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    run(a.root, a.outdir, splits=tuple(a.splits.split(',')), stat_edf=not a.no_edf_stat)


if __name__ == "__main__":
    main()
