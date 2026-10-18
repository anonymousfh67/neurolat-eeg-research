#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 2b -- Prétraitement EEG et cache de fenêtres (reprenable). train et dev UNIQUEMENT (eval est refusée).

Pour chaque enregistrement choisi :
  EDF -> 18 dérivations bipolaires (250 Hz, filtre 0,5-70 Hz, notch 60 Hz) -> fenêtres de 4 s (pas de 2 s)
  -> étiquettes de chaque fenêtre -> sélection plafonnée -> un fichier .npz (valeurs brutes en µV, float16).

Les 18 dérivations (miroir gauche/droite EXACT : on échange les indices 0-7 et 8-15 ; 16 et 17 sont symétriques) :
   0-3  FP1-F7 F7-T3 T3-T5 T5-O1      8-11  FP2-F8 F8-T4 T4-T6 T6-O2
   4-7  FP1-F3 F3-C3 C3-P3 P3-O1      12-15 FP2-F4 F4-C4 C4-P4 P4-O2
   16 FZ-CZ   17 CZ-PZ
Choix logiques (et leur raison)
  * Une dérivation n'existe que si LES DEUX électrodes existent et passent le contrôle de qualité (écart-type de
    1 à 500 µV). Pas de repli « première électrode seule » : ce serait un signal référentiel, pas bipolaire.
  * Un enregistrement avec moins de 16 dérivations valides sur 18 est rejeté et compté dans le manifeste.
  * Étiquette M1 d'une fenêtre : 1 si >= 50 % recouvre une crise de TOUT type (label != bckg), 0 si aucun
    recouvrement avec une crise, -1 (ambiguë, jamais utilisée) sinon.
  * Étiquette M2 : fenêtre entièrement contenue dans un événement labélisable (règle de l'étape 1) ; les fenêtres
    sont alignées sur le début de l'événement (le label est défini à l'onset). Au plus ev_cap fenêtres par événement,
    réparties régulièrement (la première est celle de l'onset).
  * Valeurs BRUTES stockées : la normalisation (par canal et par fenêtre, ou par fenêtre) est appliquée à
    l'entraînement, ce qui permet les ablations sans refaire le prétraitement.

Usage
    python step2b_cache_windows.py --selftest
    python step2b_cache_windows.py --root <.../edf> --step1_dir <dossier de l'étape 1 train+dev> --cache_dir <cache> --limit 5   # test de vitesse
    python step2b_cache_windows.py --root ... --step1_dir ... --cache_dir ...                                                     # exécution complète (reprenable)
    python step2b_cache_windows.py --cache_dir ... --report_only
Dépendances : numpy, pandas ; mne (seulement pour lire les EDF).
"""
import argparse, os, re, sys, time, zlib, tempfile, shutil, json, glob, io, contextlib
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_inventaire_labels import parse_annotation, extract_events, label_of, REF   # une seule règle de labels

FS, WIN, STRIDE = 250, 1000, 500
WIN_S = WIN / FS
DERIVATIONS = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
               ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"), ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
               ("FZ", "CZ"), ("CZ", "PZ")]
ELECTRODES = {"FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"}
IDX = {"left": 0, "right": 1, "bilateral": 2}
MIN_VALID = 16
FLAT_STD_UV = 0.5     # une fenêtre dont l'écart-type médian (sur les dérivations valides) est < 0,5 µV est plate : signal nul / débranché


def flat_window_mask(x, valid):
    """x : (n, 18, T) en µV ; retourne un masque booléen : True = fenêtre plate (à exclure)."""
    sd = x[:, valid, :].astype(np.float32).std(axis=2)
    return np.median(sd, axis=1) < FLAT_STD_UV


@dataclass
class Cfg:
    ev_cap: int = 16          # fenêtres M2 max par événement
    sz_per_rec: int = 40      # fenêtres de crise max par enregistrement (celles des événements comprises)
    bg_per_rec: int = 8       # fenêtres de fond par enregistrement
    seed: int = 0


# ------------------------------------------------------------------ coeur numérique (testé sans EDF)
def electrode_name(ch):
    m = re.match(r"^EEG\s+([A-Z0-9]+)-(REF|LE)$", ch.upper().strip())
    return m.group(1) if m and m.group(1) in ELECTRODES else None


def build_derivations(sig):
    """sig : {électrode: signal 1D en µV}. Retourne (18, n) et le masque des dérivations valides."""
    n = len(next(iter(sig.values())))
    good = {e for e, s in sig.items() if 1.0 < float(np.std(s)) < 500.0}
    out = np.zeros((len(DERIVATIONS), n), np.float32)
    valid = np.zeros(len(DERIVATIONS), bool)
    for i, (a, b) in enumerate(DERIVATIONS):
        if a in good and b in good:
            out[i] = sig[a] - sig[b]
            valid[i] = True
    return out, valid


def seizure_intervals(rows):
    merged = []
    for a, b in sorted((r[1], r[2]) for r in rows if r[3] != "bckg"):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def m1_labels(starts_s, merged):
    ov = np.zeros(len(starts_s))
    for a, b in merged:
        ov += np.clip(np.minimum(starts_s + WIN_S, b) - np.maximum(starts_s, a), 0, None)
    frac = ov / WIN_S
    lab = np.full(len(starts_s), -1, np.int8)
    lab[frac >= 0.5] = 1
    lab[frac == 0.0] = 0
    return lab


def candidate_starts(n_samples, events):
    grid = np.arange(0, n_samples - WIN + 1, STRIDE)
    extra = []
    for ev in events:
        s0 = int(round(ev["start"] * FS))
        e1 = min(int(round(ev["stop"] * FS)), n_samples)
        extra.append(np.arange(s0, e1 - WIN + 1, STRIDE))
    allv = np.concatenate([grid] + extra) if extra else grid
    return np.unique(allv[(allv >= 0) & (allv + WIN <= n_samples)])


def m2_labels(starts, events):
    s = starts / FS
    lab = np.full(len(s), -1, np.int8)
    eid = np.full(len(s), -1, np.int16)
    n_lab = n_short = 0
    for i, ev in enumerate(events):
        l = label_of(ev, REF)
        if l is None:
            continue
        n_lab += 1
        inside = (s >= ev["start"] - 1e-9) & (s + WIN_S <= ev["stop"] + 1e-9)
        if not inside.any():
            n_short += 1
            continue
        lab[inside] = IDX[l]
        eid[inside] = i
    return lab, eid, n_lab, n_short


def select_windows(m1, eid, cfg, rng):
    keep = np.zeros(len(m1), bool)
    for e in np.unique(eid[eid >= 0]):
        idx = np.where(eid == e)[0]
        if len(idx) > cfg.ev_cap:
            idx = idx[np.unique(np.linspace(0, len(idx) - 1, cfg.ev_cap).round().astype(int))]
        keep[idx] = True
    pos = np.where((m1 == 1) & ~keep)[0]
    need = max(0, cfg.sz_per_rec - int(((m1 == 1) & keep).sum()))
    if need and len(pos):
        keep[rng.choice(pos, min(need, len(pos)), replace=False)] = True
    neg = np.where(m1 == 0)[0]
    if len(neg):
        keep[rng.choice(neg, min(cfg.bg_per_rec, len(neg)), replace=False)] = True
    return np.where(keep)[0]


def make_windows(deriv, valid, rows, stem, cfg):
    """Coeur pur : signaux (18, n) + annotations -> dictionnaire de tableaux à écrire, ou (None, raison)."""
    if int(valid.sum()) < MIN_VALID:
        return None, f"derivations_valides={int(valid.sum())}<{MIN_VALID}", {}
    events = extract_events(rows)
    starts = candidate_starts(deriv.shape[1], events)
    if len(starts) == 0:
        return None, "enregistrement_trop_court", {}
    flat = np.array([np.median(deriv[valid, s0:s0 + WIN].std(axis=1)) < FLAT_STD_UV for s0 in starts])
    n_flat = int(flat.sum())
    starts = starts[~flat]
    if len(starts) == 0:
        return None, "fenetres_plates_uniquement", {}
    m1 = m1_labels(starts / FS, seizure_intervals(rows))
    m2, eid, n_lab, n_short = m2_labels(starts, events)
    rng = np.random.default_rng(zlib.crc32(stem.encode()) ^ cfg.seed)
    sel = select_windows(m1, eid, cfg, rng)
    x = np.stack([deriv[:, s:s + WIN] for s in starts[sel]])
    bad = ~np.isfinite(x).all(axis=(1, 2))
    if bad.any():
        sel, x = sel[~bad], x[~bad]
    x = np.clip(x, -60000, 60000).astype(np.float16)
    arrays = dict(x=x, start_sample=starts[sel].astype(np.int64), m1=m1[sel], m2=m2[sel], eid=eid[sel], valid=valid)
    info = dict(n_events=len(events), n_events_labeled=n_lab, n_events_too_short=n_short, n_nonfinite_dropped=int(bad.sum()), n_flat_dropped=n_flat)
    return arrays, "ok", info


# ------------------------------------------------------------------ lecture EDF (nécessite mne ; NON testé hors Colab)
def load_electrodes(edf_path):
    """EDF -> {électrode: signal µV à 250 Hz, filtré}. Retourne (dict, sfreq d'origine) ou (None, raison)."""
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    sfreq0 = float(raw.info["sfreq"])
    pick = {}
    for name in raw.ch_names:
        e = electrode_name(name)
        if e:
            pick[e] = name
    if len(pick) < 10:
        return None, f"electrodes_trouvees={len(pick)}"
    raw.pick(list(pick.values()))
    raw.load_data(verbose="ERROR")
    raw.filter(0.5, 70.0, verbose="ERROR")
    raw.notch_filter(60.0, verbose="ERROR")
    if abs(raw.info["sfreq"] - FS) > 1e-6:
        raw.resample(FS, verbose="ERROR")
    data = raw.get_data() * 1e6
    names = list(raw.ch_names)
    return {e: data[names.index(n)] for e, n in pick.items()}, sfreq0


# ------------------------------------------------------------------ orchestration
def index_files(root, splits):
    idx = {}
    for sp in splits:
        for dp, dirs, files in os.walk(os.path.join(root, sp)):
            dirs.sort()
            for fn in files:
                if fn.endswith(".edf") or fn.endswith(".csv"):
                    d = idx.setdefault(fn[:-4], {"split": sp})
                    d["edf" if fn.endswith(".edf") else "csv"] = os.path.join(dp, fn)
    return idx


def choose_recordings(step1_dir, splits, n_neg, seed):
    rec = pd.read_csv(os.path.join(step1_dir, "recordings.csv"))
    rec = rec[rec.split.isin(splits)]
    pos = rec[rec.n_seizure_rows_any_type > 0]
    neg = rec[rec.n_seizure_rows_any_type == 0]
    neg = neg.sample(min(n_neg, len(neg)), random_state=seed)
    sel = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)      # mélange : --limit donne un mélange
    return sel[["split", "recording", "patient"]].reset_index(drop=True), len(pos), len(neg)


MANIFEST_COLS = ["split", "recording", "patient", "status", "reason", "n_windows", "n_pos", "n_neg", "n_m2_left", "n_m2_right",
                 "n_m2_bilateral", "n_valid_deriv", "sfreq_orig", "n_events", "n_events_labeled", "n_events_too_short",
                 "n_nonfinite_dropped", "seconds"]


def append_manifest(path, row):
    """Colonnes FIXES : sinon une 1re ligne d'erreur produirait un en-tête plus court que les lignes suivantes."""
    pd.DataFrame([row]).reindex(columns=MANIFEST_COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def check_manifest_header(path):
    """Un manifeste d'une ancienne version (colonnes différentes) est mis de côté au lieu d'être corrompu."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != MANIFEST_COLS:
            old = path.replace(".csv", f"_ancien_{int(time.time())}.csv")
            os.replace(path, old)
            print(f"manifeste d'une ancienne version mis de côté : {old}", flush=True)


def process_all(root, step1_dir, cache_dir, splits, n_neg, cfg, limit=None):
    if "eval" in splits:
        raise SystemExit("La partition eval est refusée ici : elle ne sert qu'à l'évaluation finale.")
    os.makedirs(cache_dir, exist_ok=True)
    man_path = os.path.join(cache_dir, "manifest.csv")
    check_manifest_header(man_path)
    if os.path.exists(man_path):
        prev = pd.read_csv(man_path).drop_duplicates("recording", keep="last")
        done = set(prev[prev.status.isin(["ok", "rejected"])].recording)     # une erreur sera retentée
    else:
        done = set()
    sel, n_pos, n_neg_av = choose_recordings(step1_dir, splits, n_neg, cfg.seed)
    print(f"enregistrements retenus : {len(sel)} ({n_pos} avec crise + {len(sel) - n_pos} sans crise) ; déjà traités : {len(done)}", flush=True)
    idx = index_files(root, splits)
    todo = [r for r in sel.itertuples() if r.recording not in done]
    if limit:
        todo = todo[:limit]
    t_all = time.time()
    for k, r in enumerate(todo):
        t0 = time.time()
        row = dict(split=r.split, recording=r.recording, patient=r.patient, status="", reason="", n_windows=0, n_pos=0, n_neg=0,
                   n_m2_left=0, n_m2_right=0, n_m2_bilateral=0, n_valid_deriv=0, sfreq_orig=None, n_events=0,
                   n_events_labeled=0, n_events_too_short=0, n_nonfinite_dropped=0, seconds=0.0)
        try:
            f = idx.get(r.recording, {})
            if "edf" not in f or "csv" not in f:
                row.update(status="rejected", reason="fichier_edf_ou_csv_introuvable")
            else:
                _, rows = parse_annotation(f["csv"])
                sig, info0 = load_electrodes(f["edf"])
                if sig is None:
                    row.update(status="rejected", reason=info0)
                else:
                    row["sfreq_orig"] = info0
                    deriv, valid = build_derivations(sig)
                    row["n_valid_deriv"] = int(valid.sum())
                    arrays, reason, info = make_windows(deriv, valid, rows, r.recording, cfg)
                    if arrays is None:
                        row.update(status="rejected", reason=reason)
                    else:
                        out = os.path.join(cache_dir, r.split, r.recording + ".npz")
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        tmp = out + ".tmp.npz"
                        np.savez(tmp, **arrays)
                        os.replace(tmp, out)
                        m1, m2 = arrays["m1"], arrays["m2"]
                        row.update(status="ok", reason="", n_windows=len(m1), n_pos=int((m1 == 1).sum()), n_neg=int((m1 == 0).sum()),
                                   n_m2_left=int((m2 == 0).sum()), n_m2_right=int((m2 == 1).sum()), n_m2_bilateral=int((m2 == 2).sum()), **info)
        except Exception as e:
            row.update(status="error", reason=f"{type(e).__name__}: {str(e)[:120]}")
        row["seconds"] = round(time.time() - t0, 1)
        append_manifest(man_path, row)
        if (k + 1) % 10 == 0 or limit:
            print(f"  {k + 1}/{len(todo)} | {r.recording} | {row['status']} {row['reason']} | {row['seconds']} s | "
                  f"écoulé {time.time() - t_all:.0f} s", flush=True)


def report(cache_dir):
    m = pd.read_csv(os.path.join(cache_dir, "manifest.csv")).drop_duplicates("recording", keep="last")
    ok = m[m.status == "ok"]
    L = ["<<<COPY_FROM_HERE>>>", f"enregistrements traités : {len(m)} | ok : {len(ok)} | rejetés : {(m.status == 'rejected').sum()} | erreurs : {(m.status == 'error').sum()}"]
    if len(m):
        bad = m[m.status != "ok"]
        if len(bad):
            L.append("raisons de rejet/erreur : " + str(bad.reason.fillna("").astype(str).str.replace(r"\d+", "N", regex=True).value_counts().to_dict()))
        else:
            L.append("aucun rejet, aucune erreur")
        L.append(f"durée par enregistrement : médiane {m.seconds.median():.1f} s, max {m.seconds.max():.1f} s")
    if len(ok):
        L.append(f"dérivations valides (sur 18) : {ok.n_valid_deriv.value_counts().sort_index().to_dict()}")
        L.append(f"fenêtres : {int(ok.n_windows.sum())} | M1 crise {int(ok.n_pos.sum())} / fond {int(ok.n_neg.sum())} | "
                 f"M2 gauche/droite/bilatéral : {int(ok.n_m2_left.sum())}/{int(ok.n_m2_right.sum())}/{int(ok.n_m2_bilateral.sum())}")
        L.append(f"événements : {int(ok.n_events.sum())} | labélisables {int(ok.n_events_labeled.sum())} | "
                 f"trop courts pour une fenêtre de 4 s {int(ok.n_events_too_short.sum())}")
        L.append(f"patients dans le cache : {ok.patient.nunique()} | taille estimée du cache : {ok.n_windows.sum() * 18 * WIN * 2 / 1e9:.2f} Go")
    L.append("<<<COPY_TO_HERE>>>")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(cache_dir, "cache_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


def inspect_cache(cache_dir, n=5, verbose=None):
    """Contrôle de plausibilité sur n .npz tirés au hasard (graine fixe) dans tout le cache.
    Les lignes par fichier ne sont affichées que si verbose (par défaut : seulement si n <= 10)."""
    allf = sorted(glob.glob(os.path.join(cache_dir, "*", "*.npz")))
    if not allf:
        print("aucun fichier .npz dans", cache_dir)
        return
    if n < len(allf):
        pick = np.random.default_rng(0).choice(len(allf), n, replace=False)
        files = [allf[i] for i in sorted(pick)]
    else:
        files = allf
    verbose = (n <= 10) if verbose is None else verbose
    print("<<<COPY_FROM_HERE>>>")
    ratios, file_med, max_abs, n_inval = [], [], 0.0, []
    for f in files:
        z = np.load(f)
        x = z["x"].astype(np.float32)
        valid = z["valid"]
        sd = x.std(axis=2)                                   # (fenêtres, 18) : écart-type par fenêtre et dérivation, en µV
        med = np.median(sd, axis=0)
        m1, m2 = z["m1"], z["m2"]
        win_sd = sd[:, valid].mean(axis=1)                    # amplitude moyenne de chaque fenêtre
        file_med.append(float(np.median(med[valid])))
        max_abs = max(max_abs, float(np.abs(x).max()))
        n_inval.append(int((~valid).sum()))
        ratio_txt = ""
        if (m1 == 1).any() and (m1 == 0).any() and np.median(win_sd[m1 == 0]) > 0.5:
            r = float(np.median(win_sd[m1 == 1]) / np.median(win_sd[m1 == 0]))
            ratios.append(r)
            ratio_txt = f" | amplitude crise/fond : x{r:.2f}"
        if verbose:
            print(f"{os.path.basename(f)[:-4]} | fenêtres {len(m1)} | dérivations valides {int(valid.sum())}/18 | "
                  f"M1 crise/fond {int((m1 == 1).sum())}/{int((m1 == 0).sum())} | M2 G/D/B {int((m2 == 0).sum())}/{int((m2 == 1).sum())}/{int((m2 == 2).sum())}")
            print(f"    écart-type médian par dérivation valide : min {med[valid].min():.1f}, médiane {np.median(med[valid]):.1f}, max {med[valid].max():.1f} µV"
                  f" | |x| max {np.abs(x).max():.0f} µV | dérivations invalides (mises à zéro) : {np.where(~valid)[0].tolist()}{ratio_txt}")
    fm = np.array(file_med)
    print(f"{len(files)} fichiers inspectés (tirage au hasard dans {len(allf)}) | écart-type médian par fichier : min {fm.min():.1f}, "
          f"médiane {np.median(fm):.1f}, max {fm.max():.1f} µV | fichiers < 1 µV : {int((fm < 1).sum())} | > 200 µV : {int((fm > 200).sum())} "
          f"| |x| max global {max_abs:.0f} µV | fichiers avec dérivations invalides : {sum(v > 0 for v in n_inval)}")
    if ratios:
        print(f"TEST D'ALIGNEMENT : amplitude des fenêtres de crise supérieure à celle du fond dans {sum(r > 1 for r in ratios)}/{len(ratios)} enregistrements "
              f"({100 * np.mean(np.array(ratios) > 1):.1f} %, médiane des rapports x{np.median(ratios):.2f}). Si les annotations étaient décalées par rapport au signal, on s'attendrait à ~50 %.")
    print("repères : un EEG bipolaire de scalp a typiquement un écart-type de quelques µV à ~50 µV ; < 1 ou > 200 µV est suspect.")
    print("<<<COPY_TO_HERE>>>")


def quality_scan(cache_dir):
    """Parcourt TOUT le cache : fenêtres plates, amplitudes extrêmes. Écrit cache_quality.csv."""
    rows = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "*", "*.npz"))):
        z = np.load(f)
        x, valid, m1 = z["x"], z["valid"], z["m1"]
        flat = flat_window_mask(x, valid)
        mx = np.abs(x.astype(np.float32)).max(axis=(1, 2))
        rows.append(dict(recording=os.path.basename(f)[:-4], split=os.path.basename(os.path.dirname(f)), n=len(m1), n_pos=int((m1 == 1).sum()),
                         n_neg=int((m1 == 0).sum()), n_flat=int(flat.sum()), n_flat_pos=int((flat & (m1 == 1)).sum()),
                         n_flat_neg=int((flat & (m1 == 0)).sum()), n_extreme=int((mx > 5000).sum())))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(cache_dir, "cache_quality.csv"), index=False)
    N = int(d.n.sum())
    L = ["<<<COPY_FROM_HERE>>>", f"fichiers : {len(d)} | fenêtres : {N}",
         f"fenêtres PLATES (écart-type médian < {FLAT_STD_UV} µV) : {int(d.n_flat.sum())} ({100 * d.n_flat.sum() / N:.2f} %) "
         f"| parmi les fenêtres de crise : {int(d.n_flat_pos.sum())} ({100 * d.n_flat_pos.sum() / max(d.n_pos.sum(), 1):.2f} %) "
         f"| parmi les fenêtres de fond : {int(d.n_flat_neg.sum())} ({100 * d.n_flat_neg.sum() / max(d.n_neg.sum(), 1):.2f} %)",
         f"enregistrements avec au moins une fenêtre plate : {int((d.n_flat > 0).sum())} | avec plus de la moitié plates : {int((d.n_flat > d.n / 2).sum())}",
         f"fenêtres avec |x| > 5000 µV : {int(d.n_extreme.sum())} ({100 * d.n_extreme.sum() / N:.2f} %)"]
    worst = d.sort_values("n_flat", ascending=False).head(5)
    L.append("5 enregistrements les plus touchés : " + "; ".join(f"{r.recording} ({r.n_flat}/{r.n})" for r in worst.itertuples()))
    L.append("<<<COPY_TO_HERE>>>")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(cache_dir, "cache_quality_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")


# ------------------------------------------------------------------ auto-test (numpy seulement)
def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    check("électrode : 'EEG FP1-REF' -> FP1, 'EEG T3-LE' -> T3", electrode_name("EEG FP1-REF") == "FP1" and electrode_name("EEG T3-LE") == "T3")
    check("électrode : EKG, PHOTIC, A1, DC rejetées", all(electrode_name(c) is None for c in ["EEG EKG-REF", "EEG PHOTIC-REF", "EEG A1-REF", "DC1", "EEG SP1-REF"]))
    rng = np.random.default_rng(0)
    n = 250 * 120
    sig = {e: (rng.standard_normal(n) * 20).astype(np.float32) for e in ELECTRODES}
    d, v = build_derivations(sig)
    check("18 dérivations valides ; FP1-F7 = FP1 - F7", v.all() and np.allclose(d[0], sig["FP1"] - sig["F7"], atol=1e-4))
    sig2 = dict(sig); del sig2["T3"]
    d2, v2 = build_derivations(sig2)
    check("électrode absente -> dérivations T3 invalides, mises à zéro, SANS repli", (v2 == [DERIVATIONS[i][0] != "T3" and DERIVATIONS[i][1] != "T3" for i in range(18)]).all() and (d2[~v2] == 0).all())
    sig3 = dict(sig); sig3["O1"] = np.zeros(n, np.float32)
    check("électrode plate -> invalide", not build_derivations(sig3)[1][3])
    sig4 = dict(sig); sig4["O1"] = sig4["O1"] * 100
    check("électrode > 500 µV -> invalide", not build_derivations(sig4)[1][3])
    # miroir : échanger 0-7 et 8-15 est une involution ; paires symétriques
    mir = {"FP1": "FP2", "F7": "F8", "T3": "T4", "T5": "T6", "O1": "O2", "F3": "F4", "C3": "C4", "P3": "P4"}
    sym = all((mir[a], mir[b]) == DERIVATIONS[i + 8] for i, (a, b) in enumerate(DERIVATIONS[:8]))
    check("miroir exact : dérivation i (0-7) <-> i+8", sym)
    # étiquettes M1
    iv = [[10.0, 20.0]]          # crise de 10 s ; fenêtres de 4 s
    s = np.array([0.0, 6.0, 7.0, 8.0, 17.0, 18.0, 19.0, 20.0])
    got = m1_labels(s, iv).tolist()
    # 0 et 6 : aucun recouvrement -> 0 ; 7 et 19 : 25 % -> ambigu ; 8, 17, 18 : >= 50 % -> 1 ; 20 : fenêtre juste après -> 0
    check(f"M1 : fond / ambigu / crise ({got})", got == [0, 0, -1, 1, 1, 1, -1, 0])
    check("M1 : exactement 50 % de recouvrement -> crise", m1_labels(np.array([8.0]), iv)[0] == 1 and m1_labels(np.array([18.0]), iv)[0] == 1)
    check("M1 : recouvrement < 50 % -> ambigu (-1)", m1_labels(np.array([7.0]), iv)[0] == -1)
    # enregistrement synthétique complet
    rows = [("FP1-F7", 30.0, 60.0, "fnsz"), ("F7-T3", 30.0, 60.0, "fnsz"), ("FP1-F3", 90.0, 93.0, "fnsz"), ("FP1-F7", 100.0, 110.0, "tcsz")]
    cfg = Cfg()
    arr, reason, info = make_windows(d, v, rows, "rec_test", cfg)
    check(f"make_windows : ok ({reason})", arr is not None and reason == "ok")
    m1, m2, st = arr["m1"], arr["m2"], arr["start_sample"]
    check("tableaux cohérents (x, étiquettes, positions)", arr["x"].dtype == np.float16 and arr["x"].shape == (len(m1), 18, WIN) and len(st) == len(m2) == len(m1))
    check("plafond M2 : <= 16 fenêtres pour l'événement de 30 s", (m2 == 0).sum() <= cfg.ev_cap and (m2 == 0).sum() > 0)
    check("la fenêtre d'onset (début = 30 s) est conservée", 30 * FS in st[m2 == 0])
    check("événement de 3 s : trop court -> compté, aucune fenêtre M2", info["n_events_too_short"] == 1 and (m2 == 1).sum() == 0)
    check("fond : <= 8 fenêtres, toutes à 0 recouvrement", (m1 == 0).sum() <= cfg.bg_per_rec and (m1 == 0).sum() > 0)
    check("aucune fenêtre ambiguë (-1) n'est stockée", bool((m1 >= 0).all()))
    check("fenêtres de crise : <= 40 en tout", (m1 == 1).sum() <= max(cfg.sz_per_rec, int((m2 >= 0).sum())))
    a2, _, _ = make_windows(d, v, rows, "rec_test", cfg)
    check("déterministe (même enregistrement, même sélection)", np.array_equal(a2["start_sample"], st))
    # fenêtres plates : la 1re minute est nulle (amplificateur débranché) -> aucune fenêtre dans cette zone
    d_flat = d.copy()
    d_flat[:, :FS * 50] = 0
    af, rf, inf_ = make_windows(d_flat, v, rows, "rec_flat", cfg)
    check(f"fenêtres plates exclues ({inf_.get('n_flat_dropped')} écartées), aucune fenêtre dans les 50 s nulles",
          af is not None and inf_["n_flat_dropped"] > 0 and (af["start_sample"] + WIN > FS * 50).all())
    check("enregistrement entièrement plat -> rejet 'fenetres_plates_uniquement'",
          make_windows(np.zeros_like(d), v, rows, "rec_zero", cfg)[1] == "fenetres_plates_uniquement")
    xx = np.stack([np.zeros((18, WIN)), rng.standard_normal((18, WIN)) * 10]).astype(np.float16)
    check("flat_window_mask : [plate, normale] -> [True, False]", flat_window_mask(xx, v).tolist() == [True, False])
    bad_valid = v.copy(); bad_valid[:4] = False
    check("moins de 16 dérivations valides -> rejet", make_windows(d, bad_valid, rows, "r", cfg)[0] is None)
    # chaîne complète sur disque (manifeste + reprise) avec chargeur simulé
    global load_electrodes
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "edf", "train", "pa000001", "s001", "01_tcp_ar"))
        stem = "pa000001_s001_t000"
        with open(os.path.join(tmp, "edf", "train", "pa000001", "s001", "01_tcp_ar", stem + ".csv"), "w") as f:
            f.write("# duration = 120.00 secs\nchannel,start_time,stop_time,label,confidence\n")
            for ch, a, b, lab in rows:
                f.write(f"{ch},{a},{b},{lab},1.0\n")
        open(os.path.join(tmp, "edf", "train", "pa000001", "s001", "01_tcp_ar", stem + ".edf"), "wb").write(b"0")
        pd.DataFrame([dict(split="train", recording=stem, patient="pa000001", n_seizure_rows_any_type=4)]).to_csv(os.path.join(tmp, "recordings.csv"), index=False)
        calls = []
        orig = load_electrodes
        load_electrodes = lambda p: (calls.append(p) or (sig, 250.0))
        process_all(os.path.join(tmp, "edf"), tmp, os.path.join(tmp, "cache"), ["train"], 0, Cfg())
        process_all(os.path.join(tmp, "edf"), tmp, os.path.join(tmp, "cache"), ["train"], 0, Cfg())
        load_electrodes = orig
        man = pd.read_csv(os.path.join(tmp, "cache", "manifest.csv"))
        z = np.load(os.path.join(tmp, "cache", "train", stem + ".npz"))
        check("fichier .npz écrit, manifeste à 1 ligne, reprise : 2e exécution ne retraite pas", len(man) == 1 and len(calls) == 1 and z["x"].dtype == np.float16)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                report(os.path.join(tmp, "cache"))
                inspect_cache(os.path.join(tmp, "cache"), 5)
                quality_scan(os.path.join(tmp, "cache"))
            check("report() et inspect_cache() fonctionnent quand tout est ok (colonne reason vide)",
                  "aucun rejet" in buf.getvalue() and "écart-type médian" in buf.getvalue() and "TEST D'ALIGNEMENT" in buf.getvalue() and "fenêtres PLATES" in buf.getvalue())
        except Exception as e:
            check(f"report()/inspect_cache() ne plantent pas ({type(e).__name__}: {e})", False)
        def boom(p):
            raise ModuleNotFoundError("mne")
        c2 = os.path.join(tmp, "cache2")
        load_electrodes = boom
        process_all(os.path.join(tmp, "edf"), tmp, c2, ["train"], 0, Cfg())
        load_electrodes = lambda p: (sig, 250.0)
        process_all(os.path.join(tmp, "edf"), tmp, c2, ["train"], 0, Cfg())
        load_electrodes = orig
        man2 = pd.read_csv(os.path.join(c2, "manifest.csv"))
        check("une erreur est retentée à la reprise (error puis ok) ET le manifeste reste lisible", list(man2.status) == ["error", "ok"] and list(man2.columns) == MANIFEST_COLS)
        c3 = os.path.join(tmp, "cache3")
        os.makedirs(c3)
        with open(os.path.join(c3, "manifest.csv"), "w") as f:
            f.write("split,recording,patient,status\ntrain,x,y,error\n")
        process_all(os.path.join(tmp, "edf"), tmp, c3, ["train"], 0, Cfg())
        check("ancien manifeste (colonnes différentes) mis de côté puis traitement normal",
              any("ancien" in n for n in os.listdir(c3)) and list(pd.read_csv(os.path.join(c3, "manifest.csv")).columns) == MANIFEST_COLS)
        try:
            process_all(os.path.join(tmp, "edf"), tmp, os.path.join(tmp, "c2"), ["train", "eval"], 0, Cfg())
            check("partition eval refusée", False)
        except SystemExit:
            check("partition eval refusée", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    print("NB : la lecture réelle des EDF (load_electrodes, mne) n'est PAS couverte par cet auto-test.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("TUSZ_ROOT", "/content/drive/MyDrive/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf"))
    ap.add_argument("--step1_dir", help="dossier contenant recordings.csv (étape 1, train+dev)")
    ap.add_argument("--cache_dir", default="cache_fenetres")
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--n_neg_recs", type=int, default=1000, help="enregistrements SANS crise tirés au hasard (fond)")
    ap.add_argument("--sz_per_rec", type=int, default=40)
    ap.add_argument("--bg_per_rec", type=int, default=8)
    ap.add_argument("--ev_cap", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="ne traiter que N enregistrements (test de vitesse)")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--inspect", type=int, default=0, help="affiche des contrôles de plausibilité sur N fichiers .npz du cache")
    ap.add_argument("--inspect_verbose", action="store_true", help="une ou deux lignes par fichier inspecté")
    ap.add_argument("--quality_scan", action="store_true", help="parcourt tout le cache : fenêtres plates, amplitudes extrêmes")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.quality_scan:
        quality_scan(a.cache_dir)
        sys.exit(0)
    if a.report_only:
        report(a.cache_dir)
        if a.inspect:
            inspect_cache(a.cache_dir, a.inspect, True if a.inspect_verbose else None)
        sys.exit(0)
    if not a.step1_dir:
        ap.error("--step1_dir est obligatoire")
    try:
        import mne  # noqa: F401
    except ImportError:
        raise SystemExit("La bibliothèque mne est absente. Dans Colab : !pip install -q mne   puis relancez cette commande.")
    cfg = Cfg(ev_cap=a.ev_cap, sz_per_rec=a.sz_per_rec, bg_per_rec=a.bg_per_rec, seed=a.seed)
    process_all(a.root, a.step1_dir, a.cache_dir, a.splits.split(","), a.n_neg_recs, cfg, a.limit)
    report(a.cache_dir)
