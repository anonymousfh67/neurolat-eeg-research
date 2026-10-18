#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference application: EDF recording -> seizure detections (M1) + left / right / bilateral label per detection (M2).

It runs EXACTLY the pipeline evaluated in the paper (same preprocessing as step 5a, same frozen decision rule as step 5b):
  18 bipolar derivations, 250 Hz, 0.5-70 Hz, 60 Hz notch; 4 s windows every 2 s; recording rejected if < 16 of 18 derivations are valid;
  M1 = mean over 3 seeds; a window is "detected" if p(M1) >= tau and not flat; detected windows closer than merge_gap_s are merged;
  a detection is kept if it has >= n_min windows; its label = argmax of the MEAN M2 probability over its windows (M2 = mean over 3 seeds).
The rule (tau, n_min, merge_gap_s, smooth, k_lat) is read from params.json; nothing is hard-coded here.

Usage
  python neurolat_app.py --edf recording.edf --weights weights/ [--out results/] [--device cpu]
  python neurolat_app.py --selftest            # no data, no torch, no mne
In Python:  from neurolat_app import analyze_edf ; r = analyze_edf("x.edf", "weights/")

Expected content of --weights:
  params.json                              frozen decision rule (written by step 5b tune on the validation set)
  m1_const_seed{0,1,2}/m1_best.pt + m1_config.json
  m2_windowglobal_seed{0,1,2}/m2_best.pt + m2_config.json     (folder prefix = "m2_<key>_seed*", key = params["m2key"])
"""
import argparse, glob, json, os, sys, tempfile, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "src"), os.path.join(HERE, "src"), HERE):
    if os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))

CLASSES = ("left", "right", "bilateral")
DISCLAIMER = ("RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE. Do not use for diagnosis or treatment. The side (left / right / bilateral) is predicted "
              "against labels derived by an algorithmic rule from channel-level annotations; these labels were not reviewed by a clinician, and the "
              "end-to-end lateralization accuracy reported in the paper is low. Detections can be false alarms; seizures can be missed.")


# ---------------------------------------------------------------------------------------------- numpy core (testable without torch / mne)
def decide(starts, p1, flat, p2, params, fs=250, win=1000):
    """Frozen rule applied to per-window scores. Returns a list of detections (dict) sorted in time.
    p2 : (n_windows, 3) mean M2 probabilities (left, right, bilateral). Same code path as step 5b (detect + argmax of the mean M2 probability)."""
    from step5b_pipeline_eval import detect, smooth_p1
    dets = detect(starts, smooth_p1(p1, params.get("smooth", 1)), flat, params["tau"], params["n_min"], params.get("merge_gap_s", 6.0))
    out = []
    for d in dets:
        ii = d["idx"] if params.get("k_lat") is None else d["idx"][:params["k_lat"]]
        pm = p2[ii].mean(0)
        out.append(dict(start_s=d["start"], stop_s=d["stop"], duration_s=d["stop"] - d["start"], n_windows=int(len(d["idx"])),
                        p1_max=float(p1[d["idx"]].max()), p1_mean=float(p1[d["idx"]].mean()),
                        p_left=float(pm[0]), p_right=float(pm[1]), p_bilateral=float(pm[2]), label=CLASSES[int(pm.argmax())]))
    return out


def mmss(t):
    t = int(round(t))
    return f"{t // 3600:d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def table_text(dets):
    if not dets:
        return "No detection."
    L = [f"{'#':>2} {'start':>8} {'end':>8} {'dur(s)':>6} {'win':>4} {'p(seiz)max':>10}   {'label':<9} {'p(L)':>5} {'p(R)':>5} {'p(B)':>5}"]
    for i, d in enumerate(dets, 1):
        L.append(f"{i:>2} {mmss(d['start_s']):>8} {mmss(d['stop_s']):>8} {d['duration_s']:>6.0f} {d['n_windows']:>4} {d['p1_max']:>10.2f}   {d['label']:<9} "
                 f"{d['p_left']:>5.2f} {d['p_right']:>5.2f} {d['p_bilateral']:>5.2f}")
    return "\n".join(L)


def plot_timeline(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"left": "tab:blue", "right": "tab:red", "bilateral": "tab:purple"}
    t = (res["starts"] + 500) / 250.0                                    # window centre (4 s windows)
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t / 60, res["p1"], color="k", lw=0.8)
    ax[0].axhline(res["params"]["tau"], color="gray", ls="--", lw=0.8, label=f"threshold tau = {res['params']['tau']}")
    ax[0].fill_between(t / 60, 0, 1, where=res["flat"], color="lightgray", alpha=0.7, label="flat windows (never detected)")
    for d in res["detections"]:
        ax[0].axvspan(d["start_s"] / 60, d["stop_s"] / 60, color=col[d["label"]], alpha=0.25)
    ax[0].set_ylabel("p(seizure), M1"); ax[0].set_ylim(0, 1); ax[0].legend(loc="upper right", fontsize=7)
    for d in res["detections"]:
        ax[1].barh(0, (d["stop_s"] - d["start_s"]) / 60, left=d["start_s"] / 60, color=col[d["label"]], height=0.6)
    ax[1].set_yticks([]); ax[1].set_xlabel("time (min)")
    ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in col.values()], labels=list(col), ncol=3, fontsize=7, loc="upper right")
    ax[0].set_title(f"{res['recording']} - {len(res['detections'])} detection(s)", fontsize=9)
    fig.text(0.01, 0.005, "Research prototype - not a medical device", fontsize=7, color="gray")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_outputs(res, outdir):
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    stem = res["recording"]
    df = pd.DataFrame(res["detections"])
    if len(df):
        df.insert(1, "start", [mmss(x) for x in df.start_s]); df.insert(3, "end", [mmss(x) for x in df.stop_s])
    df.to_csv(os.path.join(outdir, f"{stem}_detections.csv"), index=False)
    plot_timeline(res, os.path.join(outdir, f"{stem}_timeline.png"))
    with open(os.path.join(outdir, f"{stem}_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text(res) + "\n")


def report_text(res):
    r = res
    L = [DISCLAIMER, "", f"Recording: {r['recording']} ({r['duration_s'] / 60:.1f} min, {len(r['starts'])} windows, {100 * r['flat'].mean():.1f} % flat)",
         f"Rule (frozen on the validation set): tau={r['params']['tau']}, n_min={r['params']['n_min']}, merge gap {r['params'].get('merge_gap_s', 6.0)} s, "
         f"M2 normalisation key '{r['params']['m2key']}'", "", table_text(r["detections"])]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- full pipeline (needs mne + torch)
def find_models(weights):
    with open(os.path.join(weights, "params.json")) as f:
        params = json.load(f)
    m1 = sorted(glob.glob(os.path.join(weights, "m1_*_seed*")))
    m2 = sorted(glob.glob(os.path.join(weights, f"m2_{params['m2key']}_seed*")))
    if not m1 or not m2:
        raise SystemExit(f"No model found in {weights} (expected m1_*_seed* and m2_{params['m2key']}_seed*).")
    return params, m1, m2


def analyze_edf(edf_path, weights, device="cpu", predictors=None):
    """Returns a dict (status, detections, per-window scores) or {'status': 'rejected', 'reason': ...}."""
    import step2b_cache_windows as s2b
    from step5a_score_recordings import score_recording, build_predictors
    params, m1, m2 = find_models(weights)
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    sig, info = s2b.load_electrodes(edf_path)
    if sig is None:
        return dict(status="rejected", recording=stem, reason=f"not enough EEG electrodes ({info})")
    deriv, valid = s2b.build_derivations(sig)
    if int(valid.sum()) < s2b.MIN_VALID:
        return dict(status="rejected", recording=stem, reason=f"only {int(valid.sum())} of 18 derivations pass quality control (< {s2b.MIN_VALID})")
    if deriv.shape[1] < s2b.WIN:
        return dict(status="rejected", recording=stem, reason="recording shorter than 4 s")
    if predictors is None:
        predictors = build_predictors(m1, {params["m2key"]: m2}, device)
    starts, flat, out = score_recording(deriv, valid, predictors)
    p1, p2 = out["m1"], out["m2:" + params["m2key"]]
    return dict(status="ok", recording=stem, duration_s=deriv.shape[1] / s2b.FS, starts=starts, p1=p1, p2=p2, flat=flat, params=params,
                detections=decide(starts, p1, flat, p2, params), n_valid_derivations=int(valid.sum()))


# ---------------------------------------------------------------------------------------------- JSON response for the web interface
FR = {"left": "gauche", "right": "droite", "bilateral": "bilateral"}
FR_LABEL = {"gauche": "Foyer gauche", "droite": "Foyer droit", "bilateral": "Bilatéral"}


def api_response(res, max_points=6000):
    """Plain-python dict (JSON-serialisable) for the web interface. Keeps the fields of the previous interface
    (prediction, label, votes, probs, n_fenetres, n_crises, pct_crises) and adds the per-detection table and the timeline.
    Summary label = argmax of the MEAN M2 probability over all detected windows; the per-detection labels are the evaluated output."""
    if res["status"] != "ok":
        return {"error": f"Recording rejected: {res['reason']}", "recording": res["recording"]}
    d = res["detections"]
    n_win = len(res["starts"])
    dets = [dict(start_s=x["start_s"], stop_s=x["stop_s"], duration_s=x["duration_s"], n_windows=x["n_windows"], p1_max=x["p1_max"],
                 p_left=x["p_left"], p_right=x["p_right"], p_bilateral=x["p_bilateral"], label=FR[x["label"]]) for x in d]
    votes = {"gauche": 0, "droite": 0, "bilateral": 0}
    for x in dets:
        votes[x["label"]] += x["n_windows"]
    n_det_win = int(sum(votes.values()))
    thin = max(1, int(np.ceil(n_win / max_points)))
    out = dict(status="ok", recording=res["recording"], duration_s=float(res["duration_s"]), n_fenetres=int(n_win), n_crises=n_det_win,
               pct_crises=round(100.0 * n_det_win / max(n_win, 1), 1), pct_flat=round(100.0 * float(res["flat"].mean()), 1),
               n_detections=len(dets), detections=dets, votes=votes,
               rule=dict(tau=res["params"]["tau"], n_min=res["params"]["n_min"], merge_gap_s=res["params"].get("merge_gap_s", 6.0)),
               timeline=dict(step_s=2.0 * thin, win_s=4.0, p1=[round(float(v), 3) for v in res["p1"][::thin]], flat=[int(v) for v in res["flat"][::thin]]),
               disclaimer=DISCLAIMER)
    if not dets:
        out.update(prediction="non_detecte", label="Aucune crise détectée", probs=[0.0, 0.0, 0.0], labels_differ=False)
        return out
    w = np.array([x["n_windows"] for x in dets], float)
    pm = (np.array([[x["p_left"], x["p_right"], x["p_bilateral"]] for x in dets]) * w[:, None]).sum(0) / w.sum()
    pred = FR[CLASSES[int(pm.argmax())]]
    out.update(prediction=pred, label=FR_LABEL[pred], probs=[float(v) for v in pm], labels_differ=len({x["label"] for x in dets}) > 1)
    return out


# ---------------------------------------------------------------------------------------------- self-test
def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond); print(f"  [{'OK' if cond else 'FAIL'}] {name}")

    n = 120
    starts = np.arange(n) * 500
    p1 = np.full(n, 0.05); p2 = np.tile([0.2, 0.2, 0.6], (n, 1)).astype(float); flat = np.zeros(n, bool)
    p1[10:16] = 0.95; p2[10:16] = [0.7, 0.2, 0.1]                      # left detection
    p1[60:63] = 0.9; p2[60:63] = [0.1, 0.8, 0.1]                       # right detection
    p1[90:100] = 0.99; flat[90:100] = True                             # flat windows: never detected
    p1[30] = 0.85                                                      # isolated window, n_min=2 -> dropped
    params = dict(tau=0.8, n_min=2, smooth=1, merge_gap_s=6.0, k_lat=None, m2key="windowglobal")
    d = decide(starts, p1, flat, p2, params)
    check(f"two detections kept (isolated window and flat windows dropped): {[x['label'] for x in d]}", [x["label"] for x in d] == ["left", "right"])
    check("start times in seconds", abs(d[0]["start_s"] - 20.0) < 1e-9 and abs(d[1]["start_s"] - 120.0) < 1e-9)
    check("probabilities sum to 1", all(abs(x["p_left"] + x["p_right"] + x["p_bilateral"] - 1) < 1e-9 for x in d))
    d1 = decide(starts, p1, flat, p2, dict(params, n_min=1))
    check("n_min = 1 keeps the isolated window", len(d1) == 3)
    check("no detection at tau = 1.1", decide(starts, p1, flat, p2, dict(params, tau=1.1)) == [])
    check("table text lists the labels", "left" in table_text(d) and table_text([]) == "No detection.")
    import json as _json
    res_ok = dict(status="ok", recording="demo", duration_s=240.0, starts=starts, p1=p1, p2=p2, flat=flat, params=params, detections=d)
    r = api_response(res_ok)
    _json.dumps(r)                                                                  # must be JSON-serialisable
    check("api: two detections, labels in French", [x["label"] for x in r["detections"]] == ["gauche", "droite"] and r["n_detections"] == 2)
    check("api: votes = detected windows per label", r["votes"] == {"gauche": 6, "droite": 3, "bilateral": 0} and r["n_crises"] == 9)
    check("api: labels differ flagged and probs sum to 1", r["labels_differ"] and abs(sum(r["probs"]) - 1) < 1e-9)
    check("api: timeline has one value per window", len(r["timeline"]["p1"]) == n and len(r["timeline"]["flat"]) == n)
    r0 = api_response(dict(res_ok, detections=[]))
    check("api: no detection -> non_detecte", r0["prediction"] == "non_detecte" and r0["n_detections"] == 0)
    check("api: rejected recording -> error", "error" in api_response(dict(status="rejected", recording="x", reason="only 10 of 18 derivations")))
    tmp = tempfile.mkdtemp(prefix="neurolat_selftest_")
    try:
        res = dict(status="ok", recording="demo", duration_s=240.0, starts=starts, p1=p1, p2=p2, flat=flat, params=params, detections=d)
        save_outputs(res, tmp)
        files = sorted(os.listdir(tmp))
        check(f"outputs written: {files}", files == ["demo_detections.csv", "demo_report.txt", "demo_timeline.png"])
        check("disclaimer present in the report", "NOT A MEDICAL DEVICE" in open(os.path.join(tmp, "demo_report.txt"), encoding="utf-8").read())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edf"); ap.add_argument("--weights", default=os.path.join(HERE, "weights"))
    ap.add_argument("--out", default="results"); ap.add_argument("--device", default="cpu"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.edf:
        ap.error("--edf is required")
    r = analyze_edf(a.edf, a.weights, a.device)
    if r["status"] != "ok":
        print(DISCLAIMER); print(f"\nRecording {r['recording']} rejected: {r['reason']}"); sys.exit(2)
    save_outputs(r, a.out)
    print(report_text(r)); print(f"\nFiles written in {a.out}/")


if __name__ == "__main__":
    main()
