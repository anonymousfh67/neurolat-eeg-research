#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local web API for the NeuroLat interface (Flask). One endpoint does the work:  POST /analyser  (multipart field "file" = .edf)
It runs neurolat_app.analyze_edf (the pipeline evaluated in the paper) and returns neurolat_app.api_response as JSON.

  pip install -r requirements.txt
  python app/server.py [--weights weights/] [--port 5000] [--device cpu]
Then start the interface (frontend/): npm install && npm run dev  -> http://localhost:3000

The EDF is written to a temporary file that is deleted after the analysis; nothing is stored. The server listens on localhost only.
"""
import argparse, os, sys, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import neurolat_app as core


def create_app(weights, device="cpu"):
    from flask import Flask, request, jsonify
    app = Flask(__name__)
    allowed = {"http://localhost:3000", "http://127.0.0.1:3000"}

    @app.after_request
    def cors(resp):                                                # the interface (port 3000) calls this API (port 5000)
        origin = request.headers.get("Origin", "")
        if origin in allowed:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    state = {}

    def predictors():
        if "p" not in state:                                       # lazy: the server starts instantly, the networks load on first use
            from step5a_score_recordings import build_predictors
            params, m1, m2 = core.find_models(weights)
            state["params"] = params
            state["p"] = build_predictors(m1, {params["m2key"]: m2}, device)
        return state["p"]

    @app.route("/")
    def index():
        return jsonify({"status": "NeuroLat backend", "frontend": "http://localhost:3000"})

    @app.route("/status")
    def status():
        try:
            params, m1, m2 = core.find_models(weights)
            return jsonify({"ok": True, "m1_seeds": len(m1), "m2_seeds": len(m2), "rule": {k: params.get(k) for k in ("tau", "n_min", "merge_gap_s")}})
        except SystemExit as e:
            return jsonify({"ok": False, "error": str(e)})

    @app.route("/analyser", methods=["POST"])
    def analyser():
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier reçu."})
        f = request.files["file"]
        if not f.filename.lower().endswith(".edf"):
            return jsonify({"error": "Utilisez un fichier .edf"})
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            f.save(tmp.name)
            path = tmp.name
        try:
            pred = predictors()
            res = core.analyze_edf(path, weights, device, pred)
            res["recording"] = os.path.splitext(os.path.basename(f.filename))[0]
            return jsonify(core.api_response(res))
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Erreur serveur : {type(e).__name__}: {e}"})
        finally:
            os.unlink(path)

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "..", "weights")); ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    create_app(os.path.abspath(a.weights), a.device).run(host="127.0.0.1", port=a.port, debug=False)
