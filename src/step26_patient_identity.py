#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 26 -- Contrôle « identité du patient » (annotations seules, aucun signal, aucun modèle, CPU).

Question : les AUTRES événements d'un même patient prédisent-ils le label d'un événement, mieux que si on mélange les identités de patients ?
Prédicteur (leave-one-event-out, LOEO) : classe majoritaire des autres événements du patient ; les égalités et les patients à un seul événement sont exclus.
Référence de comparaison : la même statistique après 2 000 permutations des identités de patients (hypothèse nulle : l'identité n'apporte rien).
Ce prédicteur utilise des labels vrais que nul système déployable n'a : il mesure la structure qu'un raccourci « patient » pourrait exploiter, il n'est pas un
classifieur concurrent, et il ne prouve PAS que les modèles exploitent l'identité du patient.
C'est le contrôle de la section 6.3, refait avec la règle actuelle de l'étape 1 (canaux distincts, égalité -> bilatéral, tous les événements) ; il est calculé pour
la référence et pour chaque variante de l'étape 25 (robustesse à la définition du label).

Usage :
  python step26_patient_identity.py [--root .../edf] [--splits eval dev train] [--outdir ...]
  python step26_patient_identity.py --selftest
"""
import argparse, os, sys, tempfile, shutil, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step25_label_sensitivity import load_events, analyse, CLASSES


def loeo(yi, pc, P):
    cnt = np.zeros((P, 3))
    np.add.at(cnt, (pc, yi), 1)
    C = cnt[pc].copy()
    C[np.arange(len(yi)), yi] -= 1
    mx = C.max(1)
    tie = (C == mx[:, None]).sum(1) > 1
    return C.argmax(1) == yi, C.sum(1) > 0, tie


def identity_control(labels, patient, n_perm=2000, n_boot=2000, seed=0, with_extra=True):
    """labels : tableau de 'left' / 'right' / 'bilateral' ('unlabeled' ignoré)."""
    labels, patient = np.asarray(labels), np.asarray(patient)
    m = labels != "unlabeled"
    labels, patient = labels[m], patient[m]
    yi = np.array([CLASSES.index(v) for v in labels])
    pc, _ = pd.factorize(pd.Series(patient))
    P = int(pc.max()) + 1
    rng = np.random.default_rng(seed)
    ok, elig, tie = loeo(yi, pc, P)
    use = elig & ~tie
    obs = float(ok[use].mean()) if use.any() else float("nan")
    res = dict(n_evenements=int(len(yi)), n_patients=P, n_utilises=int(use.sum()), n_egalites=int((elig & tie).sum()), n_sans_autre_evenement=int((~elig).sum()),
               n_corrects=int(ok[use].sum()), exactitude=obs, classe_majoritaire=float(max((yi == c).mean() for c in range(3))))
    null = np.full(n_perm, np.nan)
    for b in range(n_perm):
        c, e, t = loeo(yi, rng.permutation(pc), P)
        u = e & ~t
        if u.any():
            null[b] = c[u].mean()
    good = ~np.isnan(null)
    res["nul_moyenne"] = float(np.nanmean(null))
    res["nul_ic95"] = [float(np.nanpercentile(null, 2.5)), float(np.nanpercentile(null, 97.5))]
    res["p_permutation"] = float((1 + np.sum(null[good] >= obs)) / (1 + good.sum()))
    if with_extra:
        sn = np.bincount(pc, weights=(ok & use).astype(float), minlength=P)
        sd = np.bincount(pc, weights=use.astype(float), minlength=P)
        idx = rng.integers(0, P, size=(n_boot, P))
        den = sd[idx].sum(1)
        r = np.where(den > 0, sn[idx].sum(1) / np.where(den > 0, den, 1), np.nan)
        res["ic95_bootstrap_patients"] = [float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))]
        big = pd.Series(patient).value_counts().index[0]
        keep = patient != big
        if keep.sum() > 2:
            pk, _ = pd.factorize(pd.Series(patient[keep]))
            ck, ek, tk = loeo(yi[keep], pk, int(pk.max()) + 1)
            uk = ek & ~tk
            res["exactitude_sans_plus_gros_patient"] = float(ck[uk].mean()) if uk.any() else float("nan")
        res["plus_gros_patient"] = str(big)
        res["n_patients_labels_mixtes"] = int((pd.Series(labels).groupby(patient).nunique() > 1).sum())
    return res


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def run_split(root, split, n_perm, n_boot):
    events = load_events(root, split)
    _, table = analyse(events, n_boot=10)
    out = {"référence": identity_control(table["référence"].values, table.patient.values, n_perm, n_boot, with_extra=True)}
    for col in table.columns[4:]:
        if col != "référence (W = 2 s, k = 2)":
            out[col] = identity_control(table[col].values, table.patient.values, n_perm, n_boot, with_extra=False)
    return out


def report(split, res):
    r = res["référence"]
    L = [f"=== Partition {split} : {r['n_evenements']} événements, {r['n_patients']} patients ===",
         f"vote des autres événements du patient : {r['n_corrects']}/{r['n_utilises']} = {pct(r['exactitude'])} % "
         f"(IC 95 % bootstrap par patient {pct(r['ic95_bootstrap_patients'][0])}-{pct(r['ic95_bootstrap_patients'][1])} %) ; "
         f"exclus : {r['n_egalites']} égalités et {r['n_sans_autre_evenement']} événements sans autre événement du patient",
         f"hypothèse nulle (identités permutées) : {pct(r['nul_moyenne'])} % [{pct(r['nul_ic95'][0])}-{pct(r['nul_ic95'][1])}], p = {r['p_permutation']:.4f} ; "
         f"classe majoritaire : {pct(r['classe_majoritaire'])} %",
         f"sans le plus gros patient ({r.get('plus_gros_patient', '?')}) : {pct(r.get('exactitude_sans_plus_gros_patient'))} % ; "
         f"patients dont les labels sont mixtes : {r.get('n_patients_labels_mixtes', '?')}/{r['n_patients']}",
         "robustesse à la définition du label : variante | exactitude LOEO | nul moyen | p | classe majoritaire"]
    for k, v in res.items():
        if k != "référence":
            L.append(f"  {k} | {pct(v['exactitude'])} % | {pct(v['nul_moyenne'])} % | {v['p_permutation']:.4f} | {pct(v['classe_majoritaire'])} %")
    return L


def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    L, allres = ["<<<COPY_FROM_HERE>>>"], {}
    for split in a.splits:
        res = run_split(a.root, split, a.n_perm, a.n_boot)
        allres[split] = res
        L += report(split, res) + [""]
    L.append("<<<COPY_TO_HERE>>>")
    json.dump(allres, open(os.path.join(a.outdir, "identite_patient.json"), "w"), indent=2, ensure_ascii=False)
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.outdir, "rapport_identite.txt"), "w", encoding="utf-8").write(txt + "\n")


def selftest():
    from step1_inventaire_labels import _write
    tmp = tempfile.mkdtemp(prefix="step26_selftest_")
    L2, R2 = ["FP1-F7", "F7-T3"], ["FP2-F8", "F8-T4"]
    n = 0
    # 12 patients : 6 « gauche » purs, 6 « droite » purs, 4 événements chacun -> l'identité prédit parfaitement
    for p in range(12):
        for e in range(4):
            _write(tmp, "eval", f"p{p:07d}_s001_t{e:03d}", [(c, 10, 20, "fnsz") for c in (L2 if p < 6 else R2)])
            n += 1
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    res = run_split(tmp, "eval", 300, 200)["référence"]
    check(f"{n} événements, 12 patients", res["n_evenements"] == n and res["n_patients"] == 12)
    check(f"patients « purs » : l'identité prédit tout ({res['exactitude']:.2f})", res["exactitude"] == 1.0)
    check(f"permutation : le nul est proche du hasard ({res['nul_moyenne']:.2f}) et p petit ({res['p_permutation']:.4f})", res["nul_moyenne"] < 0.75 and res["p_permutation"] < 0.01)
    check("aucune égalité", res["n_egalites"] == 0)
    # patients à labels aléatoires : l'identité ne doit rien apporter
    shutil.rmtree(tmp, ignore_errors=True)
    tmp = tempfile.mkdtemp(prefix="step26_selftest_")
    rng = np.random.default_rng(0)
    for p in range(30):
        for e in range(5):
            _write(tmp, "eval", f"q{p:07d}_s001_t{e:03d}", [(c, 10, 20, "fnsz") for c in (L2 if rng.random() < 0.5 else R2)])
    r2 = run_split(tmp, "eval", 300, 200)["référence"]
    check(f"patients à labels aléatoires : exactitude proche du nul ({r2['exactitude']:.2f} contre {r2['nul_moyenne']:.2f})", abs(r2["exactitude"] - r2["nul_moyenne"]) < 0.12)
    check(f"... et p non significatif ({r2['p_permutation']:.3f})", r2["p_permutation"] > 0.05)
    out = os.path.join(tmp, "out")
    main(argparse.Namespace(root=tmp, splits=["eval"], outdir=out, n_perm=100, n_boot=100))
    check("rapport écrit", os.path.exists(os.path.join(out, "rapport_identite.txt")))
    shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("TUSZ_ROOT", "/content/drive/MyDrive/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf"))
    ap.add_argument("--splits", nargs="+", default=["eval"])
    ap.add_argument("--outdir", default="/content/drive/MyDrive/projet_propre/label_sensitivity")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest()) if a.selftest else main(a)
