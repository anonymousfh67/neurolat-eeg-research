#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 21 -- Dossier de RELECTURE CLINIQUE À L'AVEUGLE des labels (gauche / droite / bilatéral).

Objectif. Notre label vient d'une règle de comptage de canaux annotés (étape 1), jamais vérifiée par un clinicien. Ce script prépare 60 événements de TUSZ
(train + dev, JAMAIS eval), tirés par strates, présentés sous forme de tracés EEG (10 s avant -> 20 s après le début annoté), dans un ordre aléatoire, SANS
label, SANS type de crise et SANS identité de patient. Un neurologue répond « left / right / bilateral / unclear » pour chaque événement. Le script `score`
mesure ensuite l'accord entre sa lecture et notre règle (kappa de Cohen avec IC95 bootstrap, accord par strate).

Strates (tirage fixé, un événement par patient quand c'est possible) :
   S1 focal, règle = gauche (12)  |  S2 focal, règle = droite (12)  |  S3 focal, règle = bilatéral (24, suréchantillonné : c'est la zone douteuse)
   S4 généralisé seul (gnsz), règle = bilatéral (12)
Aucun événement d'eval. Le relecteur ne voit ni la strate ni la règle.

Usage :
  python step21_audit_pack.py build --events_csv .../etape1_train_dev/events.csv --root .../tuh_eeg_seizure/v2.0.3/edf --out_dir .../audit
  (envoyer audit_events.pdf + reponses_a_remplir.csv au relecteur ; garder cle_secrete.csv)
  python step21_audit_pack.py score --key .../audit/cle_secrete.csv --answers .../audit/reponses_remplies.csv
  python step21_audit_pack.py --selftest
"""
import argparse, os, sys, tempfile, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STRATA = {"S1": ("focal", "left", 12), "S2": ("focal", "right", 12), "S3": ("focal", "bilateral", 24), "S4": ("gen", "bilateral", 12)}
SPLITS_OK = ("train", "dev")
PRE_S, POST_S = 10.0, 20.0
MIN_DUR_S = 8.0
DERIV_NAMES = ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2", "Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "Fz-Cz", "Cz-Pz"]
ANSWERS = {"left": "left", "l": "left", "right": "right", "r": "right", "bilateral": "bilateral", "b": "bilateral", "unclear": "unclear", "u": "unclear"}


def stratum(has_focal, label):
    for k, (t, lab, _) in STRATA.items():
        if (t == "focal") == bool(has_focal) and lab == label:
            return k
    return None


def sample_events(events_csv, seed=0):
    ev = pd.read_csv(events_csv)
    ev["has_focal"] = ev["has_focal"].astype(str).str.lower().isin(["true", "1"])
    ev = ev[ev.split.isin(SPLITS_OK) & (ev.label.isin(["left", "right", "bilateral"]))].copy()
    ev["dur"] = ev.stop - ev.start
    ev = ev[ev.dur >= MIN_DUR_S]
    ev["stratum"] = [stratum(h, l) for h, l in zip(ev.has_focal, ev.label)]
    ev = ev[ev.stratum.notna()]
    rng = np.random.default_rng(seed)
    picks = []
    for k, (_, _, n) in STRATA.items():
        s = ev[ev.stratum == k].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        s = s.drop_duplicates("patient")                                         # un événement par patient
        if len(s) < n:
            extra = ev[(ev.stratum == k) & ~ev.index.isin(s.index)].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
            s = pd.concat([s, extra])
        picks.append(s.head(n))
    out = pd.concat(picks).sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    out["id"] = [f"E{i + 1:02d}" for i in range(len(out))]
    return out


# ------------------------------------------------------------------ tracés
def plot_event(pdf, eid, sig_deriv, t0_s, onset_s):
    """sig_deriv : (18, n) en µV à 250 Hz, démarrant à t0_s secondes de l'enregistrement ; onset_s : début annoté (s de l'enregistrement)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = 250
    n = sig_deriv.shape[1]
    t = np.arange(n) / fs + t0_s - onset_s
    scale = max(float(np.percentile(np.abs(sig_deriv), 99)), 10.0)           # même échelle pour tous les canaux : les rapports d'amplitude sont conservés
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    for i in range(sig_deriv.shape[0]):
        ax.plot(t, np.clip(sig_deriv[i], -2.5 * scale, 2.5 * scale) / scale * 0.9 - i, lw=0.5, color="black")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.text(0.1, 0.9, "annotated start", color="red", fontsize=8)
    ax.set_yticks(-np.arange(sig_deriv.shape[0]))
    ax.set_yticklabels(DERIV_NAMES[:sig_deriv.shape[0]], fontsize=8)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-sig_deriv.shape[0] + 0.3, 1.2)
    ax.set_xlabel("seconds from annotated start")
    ax.set_title(f"{eid}   (bipolar montage, 0.5-70 Hz, 60 Hz notch; common amplitude scale: 1 row = {scale * 1.1:.0f} µV)", fontsize=10)
    fig.text(0.01, 0.01, "Question: from this tracing, is the seizure onset LEFT, RIGHT or BILATERAL (generalized)? Answer in the sheet: left / right / bilateral / unclear.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def find_edf(root, split, patient, stem):
    for dp, _, files in os.walk(os.path.join(root, split, patient)):
        if stem + ".edf" in files:
            return os.path.join(dp, stem + ".edf")
    return None


def build(a):
    from matplotlib.backends.backend_pdf import PdfPages
    from step2b_cache_windows import load_electrodes, build_derivations
    sel = sample_events(a.events_csv, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    keep = []
    with PdfPages(os.path.join(a.out_dir, "audit_events.pdf")) as pdf:
        for _, r in sel.iterrows():
            edf = find_edf(a.root, r.split, r.patient, r.recording)
            if edf is None:
                print(f"  {r.id} : EDF introuvable ({r.recording}), ignoré", flush=True)
                continue
            sig, info = load_electrodes(edf)
            if sig is None:
                print(f"  {r.id} : {info}, ignoré", flush=True)
                continue
            deriv, valid = build_derivations(sig)
            t0 = max(0.0, r.start - PRE_S)
            i0, i1 = int(t0 * 250), int((r.start + POST_S) * 250)
            seg = deriv[:, i0:i1]
            if seg.shape[1] < 15 * 250 or valid.sum() < 16:
                print(f"  {r.id} : segment trop court ou dérivations invalides, ignoré", flush=True)
                continue
            plot_event(pdf, r.id, seg, t0, r.start)
            keep.append(r.id)
            print(f"  {r.id} ok", flush=True)
    sel = sel[sel.id.isin(keep)]
    sel[["id", "stratum", "recording", "patient", "event_idx", "start", "stop", "types", "label"]].to_csv(os.path.join(a.out_dir, "cle_secrete.csv"), index=False)
    pd.DataFrame(dict(id=sel.id, reviewer="", confidence_1_to_3="", comment="")).to_csv(os.path.join(a.out_dir, "reponses_a_remplir.csv"), index=False)
    print("<<<COPY_FROM_HERE>>>")
    print(f"{len(sel)} événements dans le dossier de relecture (strates : " + ", ".join(f"{k} {int((sel.stratum == k).sum())}" for k in STRATA) + ")")
    print(f"à envoyer au relecteur : {a.out_dir}/audit_events.pdf et reponses_a_remplir.csv ; à GARDER : cle_secrete.csv")
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ score
def kappa(y, p, classes=("left", "right", "bilateral")):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    if n == 0:
        return np.nan
    po = float(np.mean(y == p))
    pe = float(sum(np.mean(y == c) * np.mean(p == c) for c in classes))
    return np.nan if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def score_answers(key, ans, B=2000, seed=0):
    ans = ans.copy()
    ans["reviewer"] = ans["reviewer"].astype(str).str.strip().str.lower().map(ANSWERS)
    d = key.merge(ans[["id", "reviewer"]], on="id", how="inner")
    L = [f"{len(key)} événements dans la clé, {len(ans)} lignes de réponse, {int(d.reviewer.notna().sum())} réponses valides"]
    d_all = d[d.reviewer.notna()]
    unclear = d_all[d_all.reviewer == "unclear"]
    d = d_all[d_all.reviewer != "unclear"]
    L.append(f"« unclear » : {len(unclear)} ; événements comparés : {len(d)}")
    if len(d) == 0:
        return L, np.nan
    classes = ("left", "right", "bilateral")
    L.append("Matrice (lignes = notre règle, colonnes = relecteur) :")
    L.append(f"  {'':<10}" + "".join(f"{c:>11}" for c in classes))
    for c in classes:
        L.append(f"  {c:<10}" + "".join(f"{int(((d.label == c) & (d.reviewer == c2)).sum()):>11}" for c2 in classes))
    k = kappa(d.label, d.reviewer)
    rng = np.random.default_rng(seed)
    pats = sorted(set(d.patient))
    groups = {p: d.index[d.patient == p].values for p in pats}
    ks = []
    for _ in range(B):
        idx = np.concatenate([groups[pats[i]] for i in rng.integers(0, len(pats), len(pats))])
        v = kappa(d.loc[idx, "label"], d.loc[idx, "reviewer"])
        if not np.isnan(v):
            ks.append(v)
    lo, hi = np.percentile(ks, [2.5, 97.5]) if ks else (np.nan, np.nan)
    L.append(f"accord global : {np.mean(d.label == d.reviewer):.2f} ; kappa de Cohen {k:.2f} [{lo:.2f}, {hi:.2f}] (bootstrap par patients ; échantillon STRATIFIÉ, pas représentatif de la fréquence réelle)")
    names = {"S1": "focal, règle gauche", "S2": "focal, règle droite", "S3": "focal, règle BILATÉRAL (zone douteuse)", "S4": "généralisé seul, règle bilatéral"}
    L.append("Accord avec la règle par strate :")
    for s in STRATA:
        x = d[d.stratum == s]
        if len(x):
            dist = ", ".join(f"{c} {int((x.reviewer == c).sum())}" for c in classes)
            L.append(f"  {s} {names[s]:<40} accord {np.mean(x.label == x.reviewer):.2f} ({int((x.label == x.reviewer).sum())}/{len(x)}) | lecture du relecteur : {dist}")
    return L, k


def main_score(a):
    L, _ = score_answers(pd.read_csv(a.key), pd.read_csv(a.answers))
    print("<<<COPY_FROM_HERE>>>")
    print("\n".join(L))
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test
def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    rows = []
    for p in range(300):
        for e in range(2):
            hf = rng.random() < 0.6
            lab = rng.choice(["left", "right", "bilateral"])
            rows.append(dict(split=rng.choice(["train", "dev", "eval"]), recording=f"pat{p:03d}_s001_t00{e}", patient=f"pat{p:03d}", event_idx=0, start=30.0, stop=30.0 + rng.uniform(4, 60),
                             has_focal=hf, types="fnsz" if hf else "gnsz", label=lab))
    with tempfile.TemporaryDirectory() as tmp:
        ev = os.path.join(tmp, "events.csv")
        pd.DataFrame(rows).to_csv(ev, index=False)
        sel = sample_events(ev, seed=0)
        check(f"60 événements au total ({len(sel)})", len(sel) == 60)
        check("répartition S1/S2/S3/S4 = 12/12/24/12", [int((sel.stratum == k).sum()) for k in STRATA] == [12, 12, 24, 12])
        check("aucun événement d'eval", set(sel.split) <= set(SPLITS_OK))
        check("durée minimale respectée", ((sel.stop - sel.start) >= MIN_DUR_S).all())
        check("identifiants anonymes E01..E60 sans lien avec la strate (ordre mélangé)", list(sel.id) == [f"E{i + 1:02d}" for i in range(60)] and len(set(sel.stratum.iloc[:12])) > 1)
        check("tirage reproductible", sample_events(ev, seed=0).equals(sel))
        # tracé
        from matplotlib.backends.backend_pdf import PdfPages
        pdfp = os.path.join(tmp, "t.pdf")
        with PdfPages(pdfp) as pdf:
            plot_event(pdf, "E01", rng.standard_normal((18, 7500)) * 20, 20.0, 30.0)
        check("une page PDF est écrite", os.path.getsize(pdfp) > 2000)
        # notation : relecteur d'accord à 90 % sauf sur S3 (30 %)
        ans = []
        for _, r in sel.iterrows():
            agree = rng.random() < (0.3 if r.stratum == "S3" else 0.9)
            lab = r.label if agree else rng.choice([c for c in ["left", "right", "bilateral"] if c != r.label])
            ans.append(dict(id=r.id, reviewer=lab.upper() if rng.random() < 0.5 else lab))
        L, k = score_answers(sel, pd.DataFrame(ans), B=300)
        txt = "\n".join(L)
        s3 = sel[sel.stratum == "S3"]
        check("rapport : matrice, kappa, accord par strate", all(s in txt for s in ("Matrice", "kappa de Cohen", "S3", "S4")))
        a3 = [l for l in L if l.strip().startswith("S3")][0]
        check("l'accord faible sur S3 est visible dans le rapport", "accord 0." in a3 and float(a3.split("accord ")[1].split(" ")[0]) < 0.6)
    check("kappa = 1 pour un accord parfait", abs(kappa(["left", "right", "bilateral", "left"], ["left", "right", "bilateral", "left"]) - 1) < 1e-9)
    check("kappa ~ 0 pour des réponses sans lien", abs(kappa(["left", "left", "right", "right"], ["left", "right", "left", "right"])) < 1e-9)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    base = "/content/drive/MyDrive/"
    b.add_argument("--events_csv", default=base + "projet_propre/etape1_train_dev/events.csv")
    b.add_argument("--root", default=base + "tuh_eeg/tuh_eeg_seizure/v2.0.3/edf")
    b.add_argument("--out_dir", default=base + "projet_propre/audit")
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--key", default=base + "projet_propre/audit/cle_secrete.csv")
    s.add_argument("--answers", default=base + "projet_propre/audit/reponses_remplies.csv")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else main_score(a)
