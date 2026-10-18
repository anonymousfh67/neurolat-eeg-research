#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 2a -- Découpage des PATIENTS en entraînement / validation (train + dev réunis). Aucun EEG, quelques secondes.

Pourquoi : la partition `dev` seule est trop déséquilibrée pour valider (21 événements « droite » contre 808
« bilatéral »). On réunit donc `train` et `dev` (632 patients), puis on sépare par PATIENT :
aucun patient n'est à la fois dans l'entraînement et dans la validation. `eval` n'est jamais lue ici.

Méthode : tirage aléatoire de nombreux découpages par patient (graine fixée) ; on garde celui où chaque classe
(gauche / droite / bilatéral) a une part d'événements en validation la plus proche de val_frac, avec un minimum
de patients par classe en validation. Les patients sans événement labélisé sont répartis séparément.

Usage :
    python step2a_split_patients.py --step1_dir <dossier contenant recordings.csv et events.csv de train+dev> --outdir sortie_etape2a
    python step2a_split_patients.py --selftest
"""
import argparse, os, sys, tempfile, shutil
import numpy as np
import pandas as pd

CLASSES = ["left", "right", "bilateral"]


def make_split(rec, ev, val_frac=0.2, seed=0, n_tries=20000, min_pat_per_class=8):
    patients = sorted(rec.patient.unique())
    lab = ev[ev.label != "unlabeled"]
    cnt = (lab.groupby(["patient", "label"]).size().unstack(fill_value=0)
              .reindex(columns=CLASSES, fill_value=0).reindex(patients, fill_value=0))
    C = cnt.values
    has = C.sum(1) > 0
    A, B = np.where(has)[0], np.where(~has)[0]
    nA, nB = int(round(val_frac * len(A))), int(round(val_frac * len(B)))
    tot = C.sum(0)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_tries):
        va = rng.choice(A, nA, replace=False)
        sub = C[va]
        if ((sub > 0).sum(0) < min_pat_per_class).any():
            continue
        score = float(np.abs(sub.sum(0) / tot - val_frac).sum())
        if best is None or score < best[0]:
            best = (score, va)
    if best is None:
        raise SystemExit("Aucun découpage ne respecte le minimum de patients par classe : baissez min_pat_per_class.")
    vb = rng.choice(B, nB, replace=False) if nB else np.array([], int)
    val_idx = set(best[1].tolist()) | set(np.asarray(vb).tolist())
    out = pd.DataFrame({"patient": patients, "role": ["val" if i in val_idx else "train" for i in range(len(patients))]})
    out = out.join(cnt.reset_index(drop=True))
    return out, best[0]


def report(split, ev, score, val_frac):
    lab = ev[ev.label != "unlabeled"].merge(split[["patient", "role"]], on="patient")
    L = ["<<<COPY_FROM_HERE>>>", f"fraction visée en validation : {val_frac} | écart total aux cibles (0 = parfait) : {score:.4f}"]
    for role in ("train", "val"):
        p = split[split.role == role]
        g = lab[lab.role == role]
        c = g.label.value_counts()
        npat = g.groupby("label").patient.nunique()
        L.append(f"{role}: patients {len(p)} (dont {int((p[CLASSES].sum(axis=1) > 0).sum())} avec événements labélisés) | "
                 f"événements gauche/droite/bilatéral : {c.get('left', 0)}/{c.get('right', 0)}/{c.get('bilateral', 0)} | "
                 f"patients par classe : {npat.get('left', 0)}/{npat.get('right', 0)}/{npat.get('bilateral', 0)}")
    tot = lab.label.value_counts()
    L.append("part des événements en validation par classe : " + ", ".join(
        f"{c} {100 * (lab[(lab.role == 'val') & (lab.label == c)].shape[0]) / tot[c]:.1f} %" for c in CLASSES))
    both = set(split[split.role == "train"].patient) & set(split[split.role == "val"].patient)
    L.append(f"patients communs entre train et val : {sorted(both) if both else 'aucun (vérifié)'}")
    L.append("<<<COPY_TO_HERE>>>")
    return "\n".join(L)


def run(step1_dir, outdir, val_frac=0.2, seed=0, quiet=False):
    rec = pd.read_csv(os.path.join(step1_dir, "recordings.csv"))
    ev = pd.read_csv(os.path.join(step1_dir, "events.csv"))
    rec, ev = rec[rec.split.isin(["train", "dev"])], ev[ev.split.isin(["train", "dev"])]
    if rec.empty:
        raise SystemExit("Aucune ligne train/dev dans recordings.csv.")
    split, score = make_split(rec, ev, val_frac, seed)
    os.makedirs(outdir, exist_ok=True)
    split.to_csv(os.path.join(outdir, "split_patients.csv"), index=False)
    txt = report(split, ev, score, val_frac)
    with open(os.path.join(outdir, "split_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    if not quiet:
        print(txt)
    return split, txt


def selftest():
    rng = np.random.default_rng(1)
    n = 400
    sizes = np.minimum((rng.pareto(1.2, n) * 3).astype(int), 200)       # queue lourde, comme les vrais patients
    rows, recs = [], []
    for i in range(n):
        p = f"p{i:05d}"
        recs.append(dict(split="train" if i < 340 else "dev", recording=p + "_s001_t000", patient=p))
        probs = rng.dirichlet([2, 2, 2])
        for k in range(sizes[i]):
            rows.append(dict(split="train", patient=p, label=rng.choice(CLASSES, p=probs)))
    rec, ev = pd.DataFrame(recs), pd.DataFrame(rows)
    s1, _ = make_split(rec, ev, seed=0)
    s2, _ = make_split(rec, ev, seed=0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    check("chaque patient a exactement un rôle", s1.patient.is_unique and set(s1.role) == {"train", "val"})
    check("déterministe avec la même graine", s1.equals(s2))
    check("environ 20 % des patients en validation", abs((s1.role == "val").mean() - 0.2) < 0.02)
    lab = ev.merge(s1[["patient", "role"]], on="patient")
    shares = [((lab.role == "val") & (lab.label == c)).sum() / (lab.label == c).sum() for c in CLASSES]
    check(f"parts d'événements en validation proches de 20 % ({[round(x, 3) for x in shares]})", all(abs(x - 0.2) < 0.06 for x in shares))
    check("chaque classe a >= 8 patients en validation", all((s1[s1.role == 'val'][c] > 0).sum() >= 8 for c in CLASSES))
    tmp = tempfile.mkdtemp()
    rec.to_csv(os.path.join(tmp, "recordings.csv"), index=False)
    ev.assign(label_if_counting_rows=ev.label, has_focal=True).to_csv(os.path.join(tmp, "events.csv"), index=False)
    sp, txt = run(tmp, os.path.join(tmp, "o"), quiet=True)
    check("run() écrit le rapport et ne trouve aucun patient commun", "aucun (vérifié)" in txt)
    shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_dir", help="dossier contenant recordings.csv et events.csv (train+dev)")
    ap.add_argument("--outdir", default="sortie_etape2a")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.step1_dir:
        ap.error("--step1_dir est obligatoire")
    run(a.step1_dir, a.outdir, a.val_frac, a.seed)
