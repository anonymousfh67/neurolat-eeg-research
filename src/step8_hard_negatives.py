#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 8 -- Fenêtres « normales difficiles » pour réentraîner M1. TRAIN UNIQUEMENT (ni val, ni eval).

Constat : M1 n'a vu, dans son entraînement, que 8 fenêtres de fond TIRÉES AU HASARD par enregistrement (et 79 % des fenêtres d'entraînement sont des
crises). Il ne connaît donc presque pas les fenêtres normales qui ressemblent à une crise (artéfacts, rythmes lents...), d'où des fausses alarmes
et un seuil de décision très haut (0,8).

Ce script, pour chaque enregistrement du rôle TRAIN (jamais val, jamais eval) :
  1. fait passer l'enregistrement ENTIER dans l'ensemble M1 actuel (mêmes fenêtres que l'étape 5a),
  2. garde les fenêtres SANS crise (aucun recouvrement, et à plus de 10 s de toute crise annotée : on évite les débuts/fins mal annotés)
     que M1 trouve les plus « crise » (p >= seuil), au plus n_hard par enregistrement, sans recouvrement entre elles,
  3. ajoute n_rand fenêtres normales tirées au hasard dans le même enregistrement (diversité),
  4. écrit tout dans hardneg_dir/<split>/<enregistrement>.npz + manifest.csv (reprenable).
Ensuite : python step3_train_m1.py ... --hardneg_dir <hardneg_dir>  ajoute ces fenêtres (label 0) à l'entraînement de M1.

Usage :  python step8_hard_negatives.py --m1 D/m1_const_seed0 D/m1_const_seed1 D/m1_const_seed2 [--limit 3]   |   --selftest
"""
import argparse, os, sys, time, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step2b_cache_windows as s2b
from step2b_cache_windows import index_files, seizure_intervals, m1_labels, FS, WIN, MIN_VALID
from step1_inventaire_labels import parse_annotation
from step5a_score_recordings import score_recording

MARGIN_S = 10.0
MANIFEST_COLS = ["recording", "split", "patient", "status", "reason", "n_scored", "n_hard", "n_rand", "p_hard_mean", "seconds"]


def near_seizure(starts_s, merged, margin=MARGIN_S):
    """True si la fenêtre [s, s+4 s] est à moins de `margin` secondes d'une crise annotée (ou la recouvre)."""
    near = np.zeros(len(starts_s), bool)
    for a, b in merged:
        near |= (starts_s + WIN / FS > a - margin) & (starts_s < b + margin)
    return near


def select_windows(p1, flat, m1, starts, merged, n_hard, n_rand, thr, rng):
    """Retourne (indices des négatifs difficiles, indices des négatifs aléatoires). Pure fonction numpy."""
    ok = (m1 == 0) & ~flat & ~near_seizure(starts / FS, merged)
    cand = np.where(ok & (p1 >= thr))[0]
    hard = []
    for i in cand[np.argsort(-p1[cand])]:
        if len(hard) >= n_hard:
            break
        if all(abs(int(starts[i]) - int(starts[j])) >= WIN for j in hard):          # pas de recouvrement entre fenêtres retenues
            hard.append(int(i))
    hard = np.array(hard, int)
    rest = np.setdiff1d(np.where(ok)[0], hard)
    rest = rest[np.all(np.abs(starts[rest][:, None] - starts[hard][None, :]) >= WIN, axis=1)] if len(hard) else rest
    rand = rng.choice(rest, min(n_rand, len(rest)), replace=False) if len(rest) else np.array([], int)
    return hard, np.asarray(rand, int)


def list_train_recordings(cache_dir, split_csv):
    man = pd.read_csv(os.path.join(cache_dir, "manifest.csv")).drop_duplicates("recording", keep="last")
    man = man[man.status == "ok"]
    sp = pd.read_csv(split_csv)[["patient", "role"]]
    r = man.merge(sp, on="patient")
    r = r[r.role == "train"][["split", "recording", "patient"]]          # TRAIN seulement : jamais val, jamais eval
    return r.sort_values("recording").reset_index(drop=True)


def process_all(args, predictor):
    recs = list_train_recordings(args.cache_dir, args.split_csv)
    idx = index_files(args.root, sorted(recs.split.unique()))
    man_path = os.path.join(args.hardneg_dir, "manifest.csv")
    os.makedirs(args.hardneg_dir, exist_ok=True)
    done = set()
    if os.path.exists(man_path):
        prev = pd.read_csv(man_path).drop_duplicates("recording", keep="last")
        done = set(prev[prev.status.isin(["ok", "rejected"])].recording)
    todo = [r for r in recs.itertuples() if r.recording not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"enregistrements de train : {len(recs)}, déjà faits {len(done)}, à faire {len(todo)}", flush=True)
    t_all = time.time()
    for k, r in enumerate(todo):
        t0 = time.time()
        row = dict(recording=r.recording, split=r.split, patient=r.patient, status="", reason="", n_scored=0, n_hard=0, n_rand=0, p_hard_mean=np.nan, seconds=0.0)
        try:
            f = idx.get(r.recording, {})
            if "edf" not in f or "csv" not in f:
                row.update(status="rejected", reason="fichier_edf_ou_csv_introuvable")
            else:
                sig, info = s2b.load_electrodes(f["edf"])
                if sig is None:
                    row.update(status="rejected", reason=str(info))
                else:
                    deriv, valid = s2b.build_derivations(sig)
                    if int(valid.sum()) < MIN_VALID or deriv.shape[1] < WIN:
                        row.update(status="rejected", reason="derivations_insuffisantes_ou_trop_court")
                    else:
                        _, rows = parse_annotation(f["csv"])
                        merged = seizure_intervals(rows)
                        starts, flat, out = score_recording(deriv, valid, {"m1": predictor})
                        p1 = out["m1"]
                        m1 = m1_labels(starts / FS, merged)
                        rng = np.random.default_rng(int(args.seed) + (zlib.crc32(r.recording.encode()) % 100000))
                        hard, rand = select_windows(p1, flat, m1, starts, merged, args.n_hard, args.n_rand, args.thr, rng)
                        sel = np.concatenate([hard, rand]).astype(int)
                        if len(sel):
                            x = np.clip(np.stack([deriv[:, s:s + WIN] for s in starts[sel]]), -60000, 60000).astype(np.float16)
                            path = os.path.join(args.hardneg_dir, r.split, r.recording + ".npz")
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            np.savez(path + ".tmp.npz", x=x, start_sample=starts[sel].astype(np.int64), m1=np.zeros(len(sel), np.int8),
                                     m2=np.full(len(sel), -1, np.int8), eid=np.full(len(sel), -1, np.int16), valid=valid, p1=p1[sel].astype(np.float32),
                                     is_hard=np.r_[np.ones(len(hard), bool), np.zeros(len(rand), bool)])
                            os.replace(path + ".tmp.npz", path)
                        row.update(status="ok", n_scored=len(starts), n_hard=len(hard), n_rand=len(rand),
                                   p_hard_mean=float(p1[hard].mean()) if len(hard) else np.nan)
        except Exception as e:
            row.update(status="error", reason=f"{type(e).__name__}: {str(e)[:120]}")
        row["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame([row]).reindex(columns=MANIFEST_COLS).to_csv(man_path, mode="a", header=not os.path.exists(man_path), index=False)
        if (k + 1) % 20 == 0 or args.limit:
            print(f"  {k + 1}/{len(todo)} | {r.recording} | {row['status']} {row['reason']} | durs {row['n_hard']}+{row['n_rand']} | écoulé {time.time() - t_all:.0f} s", flush=True)
    m = pd.read_csv(man_path).drop_duplicates("recording", keep="last")
    ok = m[m.status == "ok"]
    print("<<<COPY_FROM_HERE>>>")
    print(f"enregistrements de train : ok {len(ok)} | rejetés {int((m.status == 'rejected').sum())} | erreurs {int((m.status == 'error').sum())}")
    print(f"fenêtres normales difficiles {int(ok.n_hard.sum())} (p moyenne {ok.p_hard_mean.mean():.2f}) + aléatoires {int(ok.n_rand.sum())} "
          f"| patients {ok.patient.nunique()} | taille estimée en mémoire {(ok.n_hard.sum() + ok.n_rand.sum()) * 18 * 1000 * 2 / 1e9:.1f} Go")
    print("<<<COPY_TO_HERE>>>")


def main(args):
    import torch
    from step5a_score_recordings import build_predictors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device, flush=True)
    preds = build_predictors(args.m1, {}, device)
    process_all(args, preds["m1"])


# ------------------------------------------------------------------ auto-test (numpy, sans torch ni mne)
def selftest():
    import tempfile, shutil, io, contextlib
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    rng = np.random.default_rng(0)
    # 1) sélection : crise annotée [100, 130] ; M1 très confiant sur 40 fenêtres de fond, dont 5 près de la crise
    starts = np.arange(0, 250 * 300, 500)
    s = starts / FS
    merged = [[100.0, 130.0]]
    m1 = m1_labels(s, merged)
    p1 = np.full(len(starts), 0.05)
    p1[(s >= 200) & (s < 260)] = 0.9                                  # fond loin de la crise : difficile
    p1[(s >= 92) & (s < 98)] = 0.95                                   # juste avant la crise (< 10 s) : à NE PAS retenir
    p1[(s >= 135) & (s < 139)] = 0.95                                 # juste après la crise : à NE PAS retenir
    flat = np.zeros(len(starts), bool)
    flat[(s >= 210) & (s < 214)] = True                               # plat : jamais retenu
    hard, rand = select_windows(p1, flat, m1, starts, merged, n_hard=6, n_rand=5, thr=0.3, rng=np.random.default_rng(0))
    check(f"négatifs difficiles : {len(hard)} retenus (au plus 6), tous avec p >= 0,3", 0 < len(hard) <= 6 and (p1[hard] >= 0.3).all())
    check("aucune fenêtre difficile à moins de 10 s d'une crise, aucune plate, aucune qui recouvre la crise",
          not near_seizure(s[hard], merged).any() and not flat[hard].any() and (m1[hard] == 0).all())
    check("fenêtres difficiles sans recouvrement entre elles (écart >= 4 s)", np.all(np.diff(np.sort(starts[hard])) >= WIN))
    check("fenêtres aléatoires : 5, normales, loin des crises, disjointes des difficiles",
          len(rand) == 5 and (m1[rand] == 0).all() and not near_seizure(s[rand], merged).any() and not (set(rand) & set(hard)))
    # 2) process_all sur un faux arbre : seuls les enregistrements du rôle train sont traités ; eval et val jamais lus
    tmp = tempfile.mkdtemp()
    try:
        for split, pats in (("train", ["pa000001"]), ("dev", ["pb000001"])):
            for p in pats:
                d = os.path.join(tmp, "edf", split, p, "s001", "01_tcp_ar")
                os.makedirs(d)
                open(os.path.join(d, f"{p}_s001_t000.edf"), "w").write("x")
                with open(os.path.join(d, f"{p}_s001_t000.csv"), "w") as fh:
                    fh.write("# duration = 300 secs\nchannel,start_time,stop_time,label,confidence\nFP1-F7,100.0,130.0,fnsz,1.0\nF7-T3,100.0,130.0,fnsz,1.0\n")
        cache = os.path.join(tmp, "cache")
        os.makedirs(cache)
        pd.DataFrame([dict(split="train", recording="pa000001_s001_t000", patient="pa000001", status="ok"),
                      dict(split="dev", recording="pb000001_s001_t000", patient="pb000001", status="ok")]).to_csv(os.path.join(cache, "manifest.csv"), index=False)
        pd.DataFrame([dict(patient="pa000001", role="train"), dict(patient="pb000001", role="val")]).to_csv(os.path.join(tmp, "split.csv"), index=False)
        deriv = (rng.standard_normal((18, 250 * 300)) * 15).astype(np.float32)
        valid = np.ones(18, bool)
        s2b.load_electrodes = lambda p: ({e: deriv[0] for e in s2b.ELECTRODES}, 250.0)
        s2b.build_derivations = lambda sig: (deriv, valid)
        pred = lambda x: np.where(np.abs(x.astype(np.float32)).max(axis=(1, 2)) > 0, 0.9, 0.0)[: len(x)] * 0 + 0.9 * (rng.random(len(x)) < 0.5)
        args = argparse.Namespace(root=os.path.join(tmp, "edf"), cache_dir=cache, split_csv=os.path.join(tmp, "split.csv"), hardneg_dir=os.path.join(tmp, "hn"),
                                  n_hard=8, n_rand=4, thr=0.3, seed=0, limit=None)
        with contextlib.redirect_stdout(io.StringIO()):
            process_all(args, pred)
        man = pd.read_csv(os.path.join(tmp, "hn", "manifest.csv"))
        check("seul l'enregistrement du rôle train traité (le patient val n'est jamais lu)", list(man.recording) == ["pa000001_s001_t000"] and list(man.status) == ["ok"])
        z = np.load(os.path.join(tmp, "hn", "train", "pa000001_s001_t000.npz"))
        check("npz : x (n,18,1000) float16, m1 = 0, m2 = -1, eid = -1, is_hard",
              z["x"].shape[1:] == (18, 1000) and z["x"].dtype == np.float16 and (z["m1"] == 0).all() and (z["m2"] == -1).all() and (z["eid"] == -1).all()
              and len(z["is_hard"]) == len(z["x"]))
        check("aucune fenêtre écrite ne recouvre ni n'approche la crise annotée (100-130 s, marge 10 s)",
              not near_seizure(z["start_sample"] / FS, [[100.0, 130.0]]).any())
        before = os.path.getmtime(os.path.join(tmp, "hn", "train", "pa000001_s001_t000.npz"))
        with contextlib.redirect_stdout(io.StringIO()):
            process_all(args, pred)
        check("reprise : un enregistrement déjà fait n'est pas retraité", os.path.getmtime(os.path.join(tmp, "hn", "train", "pa000001_s001_t000.npz")) == before)
        # 3) ajout à l'entraînement : append_hardneg
        from step3_train_m1 import append_hardneg
        n0 = 10
        D = dict(x=np.zeros((n0, 18, 1000), np.float16), y=np.ones(n0, np.int64), m2=np.full(n0, -1, np.int64), pid=np.zeros(n0, np.int64), rec=np.zeros(n0, np.int64),
                 start=np.arange(n0), role=np.array(["train"] * n0), valid=np.ones((n0, 18), bool), eid=np.full(n0, -1, np.int64),
                 patients=np.array(["pa000001"]), rec_names=np.array(["pa000001_s001_t000"]))
        D2 = append_hardneg(D, os.path.join(tmp, "hn"), os.path.join(tmp, "split.csv"))
        k = len(D2["y"]) - n0
        check(f"append_hardneg : {k} fenêtres ajoutées, label 0, rôle train, mêmes tailles partout",
              k == len(z["x"]) and (D2["y"][n0:] == 0).all() and (D2["role"][n0:] == "train").all()
              and all(len(D2[c]) == len(D2["y"]) for c in ("x", "m2", "pid", "rec", "start", "role", "valid", "eid")))
        # refus : un dossier contenant un patient du rôle val
        pd.DataFrame([dict(recording="pb000001_s001_t000", split="dev", patient="pb000001", status="ok")]).to_csv(os.path.join(tmp, "hn", "manifest.csv"), mode="a", header=False, index=False)
        try:
            append_hardneg(D, os.path.join(tmp, "hn"), os.path.join(tmp, "split.csv"))
            check("append_hardneg refuse un patient hors train", False)
        except SystemExit:
            check("append_hardneg refuse un patient hors train", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    print("NB : le chargement des réseaux (torch) et la lecture des EDF (mne) ne sont PAS couverts ici : lancer d'abord --limit 3 dans Colab.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("--root", default=os.environ.get("TUSZ_ROOT", "/content/drive/MyDrive/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf"))
    ap.add_argument("--cache_dir", default=base + "cache")
    ap.add_argument("--split_csv", default=base + "etape2a/split_patients.csv")
    ap.add_argument("--hardneg_dir", default=base + "hardneg")
    ap.add_argument("--m1", nargs="+", help="dossiers des entraînements de M1 actuels (une entrée par graine)")
    ap.add_argument("--n_hard", type=int, default=16, help="fenêtres normales difficiles max par enregistrement")
    ap.add_argument("--n_rand", type=int, default=6, help="fenêtres normales aléatoires supplémentaires par enregistrement")
    ap.add_argument("--thr", type=float, default=0.3, help="une fenêtre normale est « difficile » si p(M1) >= thr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.m1:
        ap.error("--m1 requis")
    main(a)
