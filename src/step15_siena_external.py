#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPE 15 -- VALIDATION EXTERNE sur Siena Scalp EEG (PhysioNet, 14 patients, 47 crises, localisation clinique par patient). AUCUN réglage, UN SEUL passage.

Le pipeline est EXACTEMENT celui gelé sur TUSZ : mêmes réseaux M1 (3 graines) et M2 windowglobal (3 graines), même règle de détection (scores/params.json),
mêmes dérivations bipolaires (reconstruites à partir des électrodes), mêmes fenêtres de 4 s / pas de 2 s. Seules adaptations techniques, inévitables :
  * le secteur est à 50 Hz en Italie (encoche à 50 Hz au lieu de 60 Hz) ; * l'EDF est à 512 Hz, rééchantillonné à 250 Hz comme pour TUSZ ;
  * noms d'électrodes de Siena (« EEG Fp1 » ...), avec T7/T8/P7/P8 rapportés à T3/T4/T5/T6.
Si l'on retouchait ensuite quoi que ce soit en regardant Siena, ce jeu ne serait plus un test : le résultat est à rapporter tel quel, une seule fois.

Mesures :
  1. détection (comme pour eval) : sensibilité, fausses alarmes/h, précision, F1, avec IC95 par patients (14 patients : intervalles larges) ;
  2. côté : pour chaque détection qui correspond à une crise annotée, le côté prédit par M2 (gauche / droite / bilatéral) est comparé à la LOCALISATION CLINIQUE
     DU PATIENT (L ou R, fichier subject_info.csv). Les patients sans côté unique ne sont pas utilisés pour cette mesure. Accord sur toutes les détections
     (« bilatéral » compté comme désaccord), accord parmi les détections G/D, et majorité par patient. Hasard : 50 % pour gauche contre droite.

Usage (Colab) :
  python step15_siena_external.py inspect --root <dossier siena>                       puis (après vérification du format)
  python step15_siena_external.py score   --root ... --m1 D/m1_const_seed0 ... --m2 windowglobal=D/m2_windowglobal_seed0,... --final_external
  python step15_siena_external.py final   --root ... [--sides PN00=L,PN01=R]
  python step15_siena_external.py --selftest
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step2b_cache_windows as s2b
from step2b_cache_windows import FS, WIN, MIN_VALID, ELECTRODES, build_derivations
import step5a_score_recordings as s5a
from step5b_pipeline_eval import detect, eval_dets, per_patient_sums, bootstrap, metrics_from_sums, COLS, PRE_TOL_S

ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
LABEL = ["gauche", "droite", "bilatéral"]
NOTCH_HZ = 50.0


# ------------------------------------------------------------------ lecture des métadonnées de Siena (formats supposés d'après la description publique ; vérifiés par `inspect`)
def siena_electrode(ch):
    c = re.sub(r"^EEG[\s_-]*", "", ch.strip(), flags=re.I).upper().strip()
    c = re.sub(r"[-_ ]?(REF|LE)$", "", c)
    c = ALIAS.get(c, c)
    return c if c in ELECTRODES else None


def _hms(s):
    """Premier horaire HH.MM.SS (ou HH:MM:SS) d'une chaîne ; espaces parasites ignorés (« 1 6.49.25 »), texte après l'heure ignoré (« 11.41.04 opure 11.40.43 »)."""
    m = re.search(r"(\d{1,2})[.:](\d{2})[.:](\d{2})", s.replace(" ", ""))
    return None if not m else int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def norm_name(fn):
    """Nom de fichier de la liste -> forme normalisée (majuscules, sans espaces, 'PNO6' -> 'PN06')."""
    fn = fn.strip().upper().replace(" ", "")
    return re.sub(r"^PNO(\d)", r"PN0\1", fn)


MAX_SZ_S = 600.0     # une « crise » plus longue que 10 min est une faute de saisie probable : durée plafonnée (signalé dans le rapport)


def parse_seizure_list(text):
    """Retourne {nom normalisé du fichier EDF: dict(reg_start (s depuis minuit), duration_txt, seizures=[(début, fin) en s depuis le début de l'enregistrement],
    capped=[...])}. Plusieurs blocs pour un même fichier sont fusionnés ; un bloc sans heure de début reprend celle du premier."""
    blocks, cur = [], None
    for line in text.splitlines():
        low = line.strip().lower()
        if ":" not in low:
            continue
        key, val = low.split(":", 1)
        key = key.strip()
        if key.startswith("file name"):
            cur = dict(name=norm_name(val), reg_start=None, reg_end=None, starts=[], ends=[])
            blocks.append(cur)
        elif cur is None:
            continue
        elif key.startswith("registration start"):
            cur["reg_start"] = _hms(val)
        elif key.startswith("registration end"):
            cur["reg_end"] = _hms(val)
        elif key.startswith("seizure start") or key.startswith("start time"):          # PN01 écrit « Start time » sans « Seizure »
            cur["starts"].append(_hms(val))
        elif key.startswith("seizure end") or key.startswith("end time"):
            cur["ends"].append(_hms(val))
    merged = {}
    for c in blocks:
        m = merged.setdefault(c["name"], dict(reg_start=None, reg_end=None, starts=[], ends=[]))
        for k in ("reg_start", "reg_end"):
            if m[k] is None:
                m[k] = c[k]
        m["starts"] += c["starts"]
        m["ends"] += c["ends"]
    res = {}
    for fn, c in merged.items():
        r0 = c["reg_start"]
        if r0 is None or len(c["starts"]) != len(c["ends"]) or any(v is None for v in c["starts"] + c["ends"]):
            raise SystemExit(f"ERREUR : liste de crises illisible pour {fn} : {c}")
        sz, capped = [], []
        for a, b in zip(c["starts"], c["ends"]):
            a_ = (a - r0) % 86400
            b_ = (b - r0) % 86400
            if b_ < a_:
                b_ += 86400
            if b_ - a_ > MAX_SZ_S:
                capped.append((float(a_), float(b_)))
                b_ = a_ + MAX_SZ_S
            sz.append((float(a_), float(b_)))
        dur = None if c["reg_end"] is None else float((c["reg_end"] - r0) % 86400)
        res[fn] = dict(reg_start=r0, duration_txt=dur, seizures=sz, capped=capped)
    return res


def assign_lists(edf_paths, sz):
    """{nom d'EDF (sans extension, majuscules): entrée de liste ou None}. Correspondance exacte d'abord ; sinon, par patient, un seul EDF restant avec une seule entrée restante
    (cas « PN01.edf », « PN11-.edf »). Retourne aussi les entrées de liste restées sans EDF."""
    by_name = {os.path.basename(p).upper(): p for p in edf_paths}
    out, used = {}, set()
    for nm, p in by_name.items():
        if nm in sz:
            out[os.path.splitext(nm)[0]] = sz[nm]
            used.add(nm)
    for pat in sorted({patient_of(n) for n in by_name}):
        rest_edf = [n for n in by_name if patient_of(n) == pat and os.path.splitext(n)[0] not in out]
        rest_sz = [n for n in sz if patient_of(n) == pat and n not in used]
        if len(rest_edf) == 1 and len(rest_sz) == 1:
            out[os.path.splitext(rest_edf[0])[0]] = sz[rest_sz[0]]
            used.add(rest_sz[0])
    for nm in by_name:
        out.setdefault(os.path.splitext(nm)[0], None)
    return out, [n for n in sz if n not in used]


def parse_sides(text):
    """subject_info.csv -> {patient: 'L' | 'R' | None}. Cherche la colonne de localisation/latéralité ; ne prend que L/R seuls (pas de supposition sur 'T')."""
    from io import StringIO
    df = pd.read_csv(StringIO(text), sep=None, engine="python")
    idcol = df.columns[0]
    cols = [c for c in df.columns if re.search(r"local|later|side|hemi|lobe", str(c), re.I)]
    sides = {}
    for _, r in df.iterrows():
        pid = str(r[idcol]).strip().upper()
        toks = set()
        for c in cols:
            toks |= {t.upper() for t in re.split(r"[^A-Za-z]+", str(r[c])) if t}
        l, rr = bool(toks & {"L", "LEFT"}), bool(toks & {"R", "RIGHT"})
        sides[pid] = "L" if l and not rr else ("R" if rr and not l else None)
    return sides


def patient_of(name):
    m = re.match(r"^(PN\d+)", os.path.basename(name).upper())
    return m.group(1) if m else os.path.splitext(os.path.basename(name))[0].upper()


def index_siena(root):
    edfs = sorted(glob.glob(os.path.join(root, "**", "*.edf"), recursive=True))
    lists = sorted(glob.glob(os.path.join(root, "**", "Seizures-list-*.txt"), recursive=True))
    sz = {}
    for p in lists:
        sz.update(parse_seizure_list(open(p, encoding="utf-8", errors="ignore").read()))
    info = glob.glob(os.path.join(root, "**", "subject_info.csv"), recursive=True)
    sides = parse_sides(open(info[0], encoding="utf-8", errors="ignore").read()) if info else {}
    return edfs, sz, sides


# ------------------------------------------------------------------ lecture EDF (mne), mêmes traitements que TUSZ sauf l'encoche
def process_channel(x, sfreq):
    """Un canal (µV, fréquence d'origine) -> filtre 0,5-70 Hz, encoche 50 Hz, rééchantillonnage à 250 Hz (mêmes fonctions mne que raw.filter / notch_filter / resample,
    appliquées canal par canal : la mémoire reste petite pour les enregistrements de 23 h)."""
    import mne
    x = mne.filter.filter_data(x.astype(np.float64), sfreq, 0.5, 70.0, verbose="ERROR")
    x = mne.filter.notch_filter(x, sfreq, [NOTCH_HZ], verbose="ERROR")
    if abs(sfreq - FS) > 1e-6:
        x = mne.filter.resample(x, up=float(FS), down=float(sfreq), verbose="ERROR")
    return x.astype(np.float32)


def load_siena_electrodes(edf_path):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    pick = {}
    for name in raw.ch_names:
        e = siena_electrode(name)
        if e and e not in pick:
            pick[e] = name
    if len(pick) < 10:
        return None, f"electrodes_trouvees={len(pick)}", float("nan")
    sig = {}
    for e, name in pick.items():
        sig[e] = process_channel(raw.get_data(picks=[name])[0] * 1e6, sfreq)
    md = raw.info.get("meas_date")
    start_s = float(md.hour * 3600 + md.minute * 60 + md.second) if md is not None else float("nan")
    return sig, sfreq, start_s


# ------------------------------------------------------------------ évaluation
def matched_labels(dets, S):
    return [d["label"] for d in dets if any(a - PRE_TOL_S <= d["start"] <= b for a, b in S)]


def side_summary(per_patient_labels, sides, B=2000, seed=0):
    """per_patient_labels : {patient: [labels 0/1/2 des détections appariées à une crise]} ; sides : {patient: 'L'|'R'|None}. Retourne des lignes de texte."""
    pats = sorted(p for p in per_patient_labels if sides.get(p) in ("L", "R"))
    if not pats:
        return ["aucun patient avec un côté clinique unique (L ou R) : mesure du côté impossible"]
    def counts(ps):
        n = c_all = n_gd = c_gd = maj_n = maj_c = 0
        nl = nr = nl_gd = cl_gd = nr_gd = cr_gd = 0
        for p in ps:
            y = 0 if sides[p] == "L" else 1
            labs = np.array(per_patient_labels[p], int)
            n += len(labs); c_all += int((labs == y).sum())
            gd = labs[labs < 2]
            n_gd += len(gd); c_gd += int((gd == y).sum())
            if y == 0:
                nl += len(labs); nl_gd += len(gd); cl_gd += int((gd == 0).sum())
            else:
                nr += len(labs); nr_gd += len(gd); cr_gd += int((gd == 1).sum())
            if len(gd) and (gd == 0).sum() != (gd == 1).sum():
                maj_n += 1; maj_c += int(int((gd == 1).sum() > (gd == 0).sum()) == y)
            elif len(labs):
                maj_n += 1               # égalité ou seulement des bilatéraux : compté comme désaccord
        return np.array([n, c_all, n_gd, c_gd, maj_n, maj_c, nl, nr, nl_gd, cl_gd, nr_gd, cr_gd], float)
    pt = np.array([counts([p]) for p in pats])
    tot = pt.sum(0)
    rng = np.random.default_rng(seed)
    bs = np.array([pt[rng.integers(0, len(pats), len(pats))].sum(0) for _ in range(B)])
    def line(name, num, den):
        with np.errstate(invalid="ignore", divide="ignore"):
            v = tot[num] / tot[den]
            lo, hi = np.nanpercentile(bs[:, num] / bs[:, den], [2.5, 97.5])
        return f"  {name:<62} {v:.3f} [{lo:.3f}, {hi:.3f}]  ({int(tot[num])}/{int(tot[den])})"
    def bal(b):
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(np.nanmean([b[9] / b[8], b[11] / b[10]]))
    bl = np.array([bal(b) for b in bs])
    lo_b, hi_b = np.nanpercentile(bl, [2.5, 97.5])
    base = max(tot[6], tot[7]) / tot[0] if tot[0] else float("nan")
    out = [f"patients avec côté clinique unique : {len(pats)} ({sum(sides[p] == 'L' for p in pats)} gauche, {sum(sides[p] == 'R' for p in pats)} droite) : le jeu est DÉSÉQUILIBRÉ, lire les lignes équilibrées",
           line("accord sur toutes les détections appariées (bilatéral = désaccord)", 1, 0),
           f"  {'référence : toujours répondre le côté majoritaire (accord sur toutes les détections)':<62} {base:.3f}",
           line("accord parmi les détections prédites gauche ou droite", 3, 2),
           line("rappel du côté GAUCHE (patients cliniquement gauche, détections G/D)", 9, 8),
           line("rappel du côté DROIT (patients cliniquement droit, détections G/D)", 11, 10),
           f"  {'exactitude équilibrée gauche/droite (moyenne des deux rappels ; hasard 0,5)':<62} {bal(tot):.3f} [{lo_b:.3f}, {hi_b:.3f}]",
           line("accord de la majorité par patient (patients avec détection)", 5, 4)]
    out.append("  par patient : " + " | ".join(f"{p} cliniq {sides[p]} -> G/D/B " + "/".join(str(int((np.array(per_patient_labels[p]) == k).sum())) for k in range(3)) for p in pats))
    return out


def evaluate(recs, params, m2key, sides, B=2000):
    """recs : [(patient, sc, S)]. Retourne les lignes de rapport."""
    tau, n_min, sm, gap, kl = params["tau"], params["n_min"], params.get("smooth", 1), params.get("merge_gap_s", 6.0), params.get("k_lat")
    recs5 = [(p, sc, S, []) for p, sc, S in recs]
    pats, M = per_patient_sums(recs5, tau, n_min, m2key, sm, kl, gap)
    point = metrics_from_sums(M.sum(0))
    boot = bootstrap(M, B)
    lines = [f"{len(pats)} patients ; {M[:, COLS.index('hours')].sum():.1f} h d'enregistrement ; crises annotées {int(M[:, 0].sum() + M[:, 1].sum())}", "Détection (règle gelée sur TUSZ) :"]
    for k in ("sensibilité", "fausses_alarmes_par_h", "précision", "F1", "part_du_temps_en_alarme"):
        lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        lines.append(f"  {k:30s} {float(point[k]):.3f}  [{lo:.3f}, {hi:.3f}]")
    per = {}
    from step5b_pipeline_eval import smooth_p1
    for p, sc, S in recs:
        dets = detect(sc["starts"], smooth_p1(sc["p1"], sm), sc["flat"], tau, n_min, gap)
        eval_dets(sc, S, [], dets, m2key, kl)
        per.setdefault(p, []).extend(matched_labels(dets, S))
    lines.append("Côté prédit par M2 comparé à la localisation clinique du patient :")
    lines += side_summary(per, sides, B)
    return lines


# ------------------------------------------------------------------ commandes
def cmd_inspect(a):
    edfs, sz, sides = index_siena(a.root)
    assign, orphan = assign_lists(edfs, sz)
    print("<<<COPY_FROM_HERE>>>")
    print(f"{len(edfs)} fichiers EDF ; {len(sz)} entrées dans les listes de crises ; {sum(len(v['seizures']) for v in sz.values())} crises listées (47 attendues pour tout Siena) ; patients avec côté L/R lu : "
          f"{ {k: v for k, v in sides.items() if v} }")
    print(f"crises rattachées aux EDF présents : {sum(len(v['seizures']) for v in assign.values() if v)} ; EDF sans entrée de liste (supposés sans crise) : {sorted(k for k, v in assign.items() if v is None)} ; "
          f"entrées de liste sans EDF présent : {orphan}")
    if edfs:
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(edfs[0], preload=False, verbose="ERROR")
        print(f"premier EDF {os.path.basename(edfs[0])} : sfreq {raw.info['sfreq']} Hz ; durée {raw.n_times / raw.info['sfreq']:.0f} s ; canaux {raw.ch_names}")
        print("électrodes reconnues :", sorted({siena_electrode(c) for c in raw.ch_names if siena_electrode(c)}))
    lists = sorted(glob.glob(os.path.join(a.root, "**", "Seizures-list-*.txt"), recursive=True))
    if lists:
        print(f"--- début de {os.path.basename(lists[0])} ---")
        print("".join(open(lists[0], encoding="utf-8", errors="ignore").readlines()[:14]))
    info = glob.glob(os.path.join(a.root, "**", "subject_info.csv"), recursive=True)
    if info:
        print("--- subject_info.csv ---")
        print(open(info[0], encoding="utf-8", errors="ignore").read()[:1500])
    print("<<<COPY_TO_HERE>>>")


def cmd_score(a):
    if not a.final_external:
        raise SystemExit("ERREUR : validation externe = UN SEUL passage avec le pipeline gelé : ajouter --final_external.")
    if not os.path.exists(a.params):
        raise SystemExit("ERREUR : règle gelée introuvable (scores/params.json).")
    edfs, sz, _ = index_siena(a.root)
    if not edfs:
        raise SystemExit("ERREUR : aucun EDF sous --root.")
    m2_sets = s5a.parse_m2(a.m2)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device}", flush=True)
    predictors = s5a.build_predictors(a.m1, m2_sets, device)
    os.makedirs(os.path.join(a.out_dir, "siena"), exist_ok=True)
    with open(os.path.join(a.out_dir, "external_run_log.txt"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} validation externe Siena ; m1={a.m1} ; m2={m2_sets}\n")
    man_path = os.path.join(a.out_dir, "siena_manifest.csv")
    done = set(pd.read_csv(man_path).recording) if os.path.exists(man_path) else set()
    for k, edf in enumerate(edfs):
        stem = os.path.splitext(os.path.basename(edf))[0]
        if stem in done:
            continue
        t0 = time.time()
        row = dict(recording=stem, patient=patient_of(edf), status="", reason="", n_windows=0, duration_s=np.nan, edf_start_s=np.nan, score_path="", seconds=0.0)
        try:
            sig, info, edf_start = load_siena_electrodes(edf)
            row["edf_start_s"] = edf_start
            if sig is None:
                row.update(status="rejected", reason=info)
            else:
                deriv, valid = build_derivations(sig)
                row["duration_s"] = deriv.shape[1] / FS
                if int(valid.sum()) < MIN_VALID or deriv.shape[1] < WIN:
                    row.update(status="rejected", reason=f"derivations_valides={int(valid.sum())}")
                else:
                    starts, flat, out = s5a.score_recording(deriv, valid, predictors)
                    arrays = dict(starts=starts, p1=out["m1"], flat=flat, n_samples=deriv.shape[1])
                    arrays.update({"p2_" + kk[3:]: v for kk, v in out.items() if kk.startswith("m2:")})
                    path = os.path.join(a.out_dir, "siena", stem + ".npz")
                    np.savez(path, **arrays)
                    row.update(status="ok", n_windows=len(starts), score_path=path)
        except Exception as e:
            row.update(status="error", reason=f"{type(e).__name__}: {str(e)[:120]}")
        row["seconds"] = round(time.time() - t0, 1)
        pd.DataFrame([row]).to_csv(man_path, mode="a", header=not os.path.exists(man_path), index=False)
        print(f"  {k + 1}/{len(edfs)} | {stem} | {row['status']} {row['reason']} | {row['seconds']} s", flush=True)
    m = pd.read_csv(man_path)
    print("<<<COPY_FROM_HERE>>>")
    print(f"Siena : {len(m)} enregistrements traités | ok {int((m.status == 'ok').sum())} | rejetés {int((m.status == 'rejected').sum())} | erreurs {int((m.status == 'error').sum())}")
    print("<<<COPY_TO_HERE>>>")


TIME_TOL_S = 120.0      # écart toléré entre l'heure de début de l'EDF et celle de la liste, et entre leurs durées


def circ_diff(a, b):
    d = abs(a - b) % 86400
    return min(d, 86400 - d)


def cmd_final(a):
    with open(a.params) as f:
        P = json.load(f)
    m2key = a.m2key or P["m2key"]
    _, sz, sides = index_siena(a.root)
    if not sz or not sides:
        raise SystemExit(f"ERREUR : listes de crises ou subject_info.csv introuvables sous {a.root} (dossier effacé par une session recyclée ?). "
                         "Retélécharger seulement les métadonnées : aws s3 sync ... --exclude '*' --include '*.txt' --include '*.csv'")
    if a.sides:
        for kv in a.sides.split(","):
            p, s_ = kv.split("=")
            sides[p.strip().upper()] = s_.strip().upper()
    man = pd.read_csv(os.path.join(a.out_dir, "siena_manifest.csv")).drop_duplicates("recording", keep="last")
    assign, orphan = assign_lists([os.path.join(a.root, r + ".edf") for r in man.recording], sz)      # noms issus du manifeste : les EDF ne sont pas nécessaires
    n_exp = sum(len(v["seizures"]) for v in sz.values())
    if len(man) < 41:
        print(f"ATTENTION : le manifeste ne contient que {len(man)} enregistrements sur 41 attendus : relancer `score` (reprenable) avant de conclure.", flush=True)
    recs, notes, lost, excluded = [], [], 0, []
    for r in man.itertuples():
        ent = assign.get(r.recording.upper())
        S = ent["seizures"] if ent else []
        if r.status != "ok":
            lost += len(S)
            continue
        if S:
            bad = []
            if not np.isnan(getattr(r, "edf_start_s", np.nan)) and circ_diff(r.edf_start_s, ent["reg_start"]) > TIME_TOL_S:
                bad.append(f"heure de début EDF {r.edf_start_s:.0f} s != liste {ent['reg_start']:.0f} s")
            if ent["duration_txt"] is not None and abs(ent["duration_txt"] - r.duration_s) > TIME_TOL_S:
                bad.append(f"durée EDF {r.duration_s:.0f} s != liste {ent['duration_txt']:.0f} s")
            if bad:
                excluded.append((r.recording, len(S), "; ".join(bad)))
                continue
            for c in ent["capped"]:
                notes.append(f"{r.recording}: crise annotée de {c[1] - c[0]:.0f} s (probable faute de saisie) : plafonnée à {MAX_SZ_S:.0f} s")
        z = np.load(r.score_path)
        sc = dict(starts=z["starts"], p1=z["p1"], flat=z["flat"], n_samples=int(z["n_samples"]), **{"p2_" + m2key: z["p2_" + m2key]})
        recs.append((r.patient, sc, S))
    lines = evaluate(recs, P, m2key, sides)
    print("<<<COPY_FROM_HERE>>>")
    print(f"règle FIGÉE (TUSZ) : tau={P['tau']}, n_min={P['n_min']}, fusion {P.get('merge_gap_s')} s, M2 = {m2key}")
    print(f"enregistrements dans le manifeste : {len(man)} (41 attendus) ; crises dans les listes : {n_exp} (47 attendues) ; enregistrements évalués : {len(recs)}")
    print("\n".join(lines))
    if lost:
        print(f"crises dans des enregistrements non traités : {lost}")
    for name, n_sz, why in excluded:
        print(f"EXCLU (calage temporel incertain) : {name} ({n_sz} crises) : {why}")
    for n in notes:
        print("ATTENTION", n)
    if orphan:
        print("entrées de liste sans EDF correspondant (non téléchargé ou nom différent) :", orphan)
    print("<<<COPY_TO_HERE>>>")


# ------------------------------------------------------------------ auto-test (numpy, sans mne ni torch)
def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK' if cond else 'ECHEC'}] {name}")

    check("électrodes : 'EEG Fp1' -> FP1, 'EEG T7' -> T3, 'EEG P8' -> T6, 'EKG EKG' et 'EEG Fc1' ignorés",
          siena_electrode("EEG Fp1") == "FP1" and siena_electrode("EEG T7") == "T3" and siena_electrode("EEG P8") == "T6" and siena_electrode("EKG EKG") is None and siena_electrode("EEG Fc1") is None)
    txt = ("File name: PN00-1.edf\nNumber of seizures in the file: 1\nRegistration start time: 19.39.33\nRegistration end time: 20.17.33\n"
           "Seizure start time: 19.58.36\nSeizure end time: 19.59.46\n\nFile name: PN00-2.edf\nRegistration start time: 23.50.00\nRegistration end time: 01.10.00\n"
           "Seizure start time: 00.05.00\nSeizure end time: 00.06.10\nSeizure start time: 00.30.00\nSeizure end time: 00.31.00\n")
    r = parse_seizure_list(txt)
    check("liste de crises : heures relatives au début, 1 puis 2 crises", r["PN00-1.EDF"]["seizures"] == [(1143.0, 1213.0)] and len(r["PN00-2.EDF"]["seizures"]) == 2)
    check("minuit géré : début 23.50.00, crise à 00.05.00 -> 900 s ; durée 80 min", r["PN00-2.EDF"]["seizures"][0] == (900.0, 970.0) and r["PN00-2.EDF"]["duration_txt"] == 4800.0)
    try:
        parse_seizure_list("File name: x.edf\nRegistration start time: 10.00.00\nSeizure start time: 10.10.00\n")
        check("liste illisible (début sans fin) refusée", False)
    except SystemExit:
        check("liste illisible (début sans fin) refusée", True)
    # irrégularités réelles des listes de Siena
    real = ("File name: PNO6-1.edf\nRegistration start time: 04.21.22\nRegistration end time: 07.01.37\nSeizure start time: 05.54.25\nSeizure end time: 05.55.29\n"
            "File name: PN10-4.5.6.edf\nRegistration start time: 12.11.21\nRegistration end time: 16.48.49\nSeizure start time: 12.49.50 \nSeizure end time: 12.49.55\n"
            "File name: PN10-4.5.6.edf\nRegistration start time: 12.11.21\nRegistration end time: 16.48.49\nSeizure start time: 15.18.26 (CLINICAL ONSET)\nSeizure end time: 15.19.23\n"
            "File name: PN10-7.8.9.edf\nRegistration start time:1 6.49.25\nRegistration end time: 20.58.40\nSeizure start time: 17.35.13\nSeizure end time: 17.36.01 opure 17.35.43\n"
            "File name: PN12-1.2.edf\nRegistration start time: 15.51.31\nRegistration end time: 18.34.25\nSeizure start time: 16:13.23 \nSeizure end time: 16.14.26\n"
            "File name: PN12-1.2.edf\nSeizure start time: 18.31.01 \nSeizure end time: 18.32.09\n"
            "File name: PN00-3.edf\nRegistration start time: 18.15.44\nRegistration end time: 18.57.13\nSeizure start time: 18.28.29\nSeizure end time: 19.29.29\n"
            "File name: PN11-.edf\nRegistration start time: 11.31.25 \nRegistration end time: 13.56.02\nSeizure start time: 13.37.19\nSeizure end time: 13.38.14\n")
    pn01 = ("PN01\n\nData Sampling Rate: 512 Hz\n\nChannels in EDF files:\nChannel 1: Fp1\n\nFile name: PN01.edf\nRegistration start time: 19.00.44\nRegistration end time: 08.29.41\n\n"
            "Seizure n 1 \nStart time: 21:51:02\nEnd time: 21.51.56\n\nSeizure n 2 \nStart time: 07.53.17\nEnd time: 07.54.31")
    r01 = parse_seizure_list(pn01)
    check("format PN01 (« Start time » / « End time », 'Seizure n 1') : 2 crises lues, heures après minuit gérées",
          len(r01["PN01.EDF"]["seizures"]) == 2 and r01["PN01.EDF"]["seizures"][0] == (2 * 3600 + 50 * 60 + 18.0, 2 * 3600 + 51 * 60 + 12.0) and r01["PN01.EDF"]["seizures"][1][0] == 12 * 3600 + 52 * 60 + 33)
    rr = parse_seizure_list(real)
    check("'PNO6' (lettre O) normalisé en 'PN06' ; blocs d'un même fichier fusionnés (PN10-4.5.6 : 2 crises) ; heure '16:13.23' lue",
          "PN06-1.EDF" in rr and len(rr["PN10-4.5.6.EDF"]["seizures"]) == 2 and rr["PN12-1.2.EDF"]["seizures"][0][0] == (16 * 3600 + 13 * 60 + 23) - (15 * 3600 + 51 * 60 + 31))
    check("bloc sans heure de début (PN12-1.2, 2e bloc) : reprend celle du premier ; '1 6.49.25' lu 16.49.25 ; texte après l'heure ignoré",
          len(rr["PN12-1.2.EDF"]["seizures"]) == 2 and rr["PN10-7.8.9.EDF"]["reg_start"] == 16 * 3600 + 49 * 60 + 25 and rr["PN10-7.8.9.EDF"]["seizures"][0][1] == (17 * 3600 + 36 * 60 + 1) - (16 * 3600 + 49 * 60 + 25))
    check("crise de 61 min (faute de saisie probable) plafonnée à 600 s et signalée", rr["PN00-3.EDF"]["seizures"][0][1] - rr["PN00-3.EDF"]["seizures"][0][0] == 600.0 and len(rr["PN00-3.EDF"]["capped"]) == 1)
    ass, orph = assign_lists(["/s/PN06/PN06-1.edf", "/s/PN11/PN11-1.edf", "/s/PN12/PN12-1.2.edf", "/s/PN12/PN12-3.edf", "/s/PN00/PN00-3.edf"], rr)
    check("EDF 'PN11-1' rattaché à l'entrée 'PN11-.edf' (seul restant du patient) ; 'PN12-3' sans entrée -> None ; exacts rattachés",
          ass["PN11-1"] is not None and ass["PN12-3"] is None and ass["PN06-1"] is not None and ass["PN12-1.2"] is not None)
    check("entrées de liste sans EDF signalées (PN10 : fichiers non fournis ici)", any(n.startswith("PN10") for n in orph))
    check("écart circulaire de l'heure de début géré (23.59.50 vs 00.00.10 = 20 s)", circ_diff(86390, 10) == 20)
    sides = parse_sides("Patient ID,Age,Seizure classification,Localization,Lateralization\nPN00,55,IAS,T,R\nPN01,30,WIAS,T,L\nPN02,40,FBTC,T,\nPN03,20,IAS,F,R L\n")
    check("côtés cliniques : PN00 R, PN01 L, PN02 aucun, PN03 ambigu -> aucun", sides["PN00"] == "R" and sides["PN01"] == "L" and sides["PN02"] is None and sides["PN03"] is None)
    check("patient depuis le nom de fichier", patient_of("/x/PN07/PN07-3.edf") == "PN07")
    try:
        import mne  # noqa: F401
        t = np.arange(512 * 120) / 512.0
        x = 20 * np.sin(2 * np.pi * 10 * t) + 30 * np.sin(2 * np.pi * 50 * t)
        y = process_channel(x, 512.0)
        mid = y[250 * 20:250 * 100]
        f = np.fft.rfftfreq(len(mid), 1 / 250.0)
        P = np.abs(np.fft.rfft(mid)) ** 2
        p10, p50 = P[np.argmin(abs(f - 10))], P[np.argmin(abs(f - 50))]
        check(f"canal : 512 -> 250 Hz ({len(y)} = {250 * 120} échantillons), 10 Hz conservé, 50 Hz supprimé (rapport {p10 / max(p50, 1e-9):.0f} > 100)",
              len(y) == 250 * 120 and p10 > 100 * p50)
    except ImportError:
        print("  [--] mne absent : test du traitement canal ignoré ici (il sera exécuté sur Colab si mne est installé)")
    # pipeline sur scores synthétiques : 4 patients, crises gauche ou droite, prédicteurs simples
    rng = np.random.default_rng(0)
    recs, truth = [], {}
    for i, p in enumerate(["PN00", "PN01", "PN02", "PN03"]):
        side = "L" if i % 2 == 0 else "R"
        truth[p] = side
        n = 3600 * FS
        deriv = rng.standard_normal((18, n)).astype(np.float32) * 5
        valid = np.ones(18, bool)
        a, b = 1000, 1060
        cols = slice(0, 8) if side == "L" else slice(8, 16)
        deriv[cols, a * FS:b * FS] *= 6
        preds = {"m1": lambda x: (x.astype(np.float32).std(axis=(1, 2)) > 8).astype(np.float32),
                 "m2:windowglobal": lambda x: np.stack([(x[:, :8].astype(np.float32).std(axis=(1, 2)) > x[:, 8:16].astype(np.float32).std(axis=(1, 2))).astype(float),
                                                          (x[:, 8:16].astype(np.float32).std(axis=(1, 2)) >= x[:, :8].astype(np.float32).std(axis=(1, 2))).astype(float),
                                                          np.zeros(len(x))], axis=1)}
        starts, flat, out = s5a.score_recording(deriv, valid, preds)
        sc = dict(starts=starts, p1=out["m1"], flat=flat, n_samples=n, p2_windowglobal=out["m2:windowglobal"])
        recs.append((p, sc, [(a, b)]))
    lines = evaluate(recs, dict(tau=0.5, n_min=1, smooth=1, merge_gap_s=6.0, k_lat=None), "windowglobal", truth, B=200)
    txt = "\n".join(lines)
    check("détection parfaite sur le jeu synthétique (sensibilité 1,000, IC présent)", "sensibilité                    1.000" in txt)
    check("côté : accord parfait avec la localisation clinique (1,000) et 4 patients", "patients avec côté clinique unique : 4" in txt and "(" in txt and txt.count("1.000 [1.000, 1.000]  (4/4)") == 3)
    check("un patient sans côté clinique unique est exclu de la mesure du côté", "patients avec côté clinique unique : 3" in "\n".join(evaluate(recs, dict(tau=0.5, n_min=1, smooth=1, merge_gap_s=6.0, k_lat=None), "windowglobal", {**truth, "PN03": None}, B=50)))
    print("AUTO-TEST :", "RÉUSSI" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = "/content/drive/MyDrive/projet_propre/"
    ap.add_argument("cmd", nargs="?", choices=["inspect", "score", "final"])
    ap.add_argument("--root", default="/content/siena")
    ap.add_argument("--out_dir", default=base + "scores_siena")
    ap.add_argument("--params", default=base + "scores/params.json")
    ap.add_argument("--m1", nargs="+", default=[])
    ap.add_argument("--m2", nargs="+", default=[])
    ap.add_argument("--m2key", default=None)
    ap.add_argument("--sides", default=None, help="forcer les côtés cliniques, ex. PN00=L,PN01=R")
    ap.add_argument("--final_external", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.cmd:
        ap.error("commande requise : inspect | score | final")
    {"inspect": cmd_inspect, "score": cmd_score, "final": cmd_final}[a.cmd](a)
