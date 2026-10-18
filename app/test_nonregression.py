#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-regression test: the app must give the SAME scores and detections as the evaluated pipeline (step 5a scores + step 5b rule).
For N eval recordings already scored (scores/eval/<rec>.npz), the EDF is re-processed by neurolat_app.analyze_edf and compared:
  - p(M1) and M2 probabilities identical up to float tolerance (default 1e-3: float16 input, GPU/CPU differences),
  - the list of detections (start, stop, label) identical.
Needs the EDF files, the weights and mne + torch. Reads only scores already computed; trains and tunes nothing.

  python test_nonregression.py --edf_root /path/to/tusz/edf --scores /path/to/scores --manifest /path/to/scores/scores_manifest.csv \
         --weights weights/ --n 5 --device cpu
"""
import argparse, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import neurolat_app as app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf_root", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=5); ap.add_argument("--device", default="cpu"); ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--params", default=None, help="params.json of the stored scores (default: <weights>/params.json)")
    a = ap.parse_args()
    from step2b_cache_windows import index_files
    from step5a_score_recordings import build_predictors
    params, m1, m2 = app.find_models(a.weights)
    if a.params:
        import json
        params = json.load(open(a.params))
    m = pd.read_csv(a.manifest)
    m = m[(m.status == "ok") & (m.role == "eval")]
    # choose recordings with and without detections: first n by name (deterministic)
    m = m.sort_values("recording").iloc[:: max(1, len(m) // a.n)].head(a.n)
    idx = index_files(a.edf_root, ["eval", "dev", "train"])
    predictors = build_predictors(m1, {params["m2key"]: m2}, a.device)
    bad = 0
    for r in m.itertuples():
        z = np.load(r.score_path)
        res = app.analyze_edf(idx[r.recording]["edf"], a.weights, a.device, predictors)
        d1 = np.abs(res["p1"] - z["p1"]).max(); d2 = np.abs(res["p2"] - z["p2_" + params["m2key"]]).max()
        ref = app.decide(z["starts"], z["p1"], z["flat"], z["p2_" + params["m2key"]], params)
        same = [(round(x["start_s"], 3), round(x["stop_s"], 3), x["label"]) for x in res["detections"]] == [(round(x["start_s"], 3), round(x["stop_s"], 3), x["label"]) for x in ref]
        ok = d1 < a.tol and d2 < a.tol and same and np.array_equal(res["flat"], z["flat"])
        bad += (not ok)
        print(f"{r.recording}: max|dp1|={d1:.1e} max|dp2|={d2:.1e} detections identical={same} -> {'OK' if ok else 'FAIL'}")
    print("NON-REGRESSION:", "PASSED" if bad == 0 else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
