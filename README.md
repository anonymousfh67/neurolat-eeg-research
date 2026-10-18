# Seizure detection and lateralization on TUSZ v2.0.3 - code

Code accompanying the manuscript (anonymous submission). It reproduces, from the raw corpus, every number of the paper:
label definition, patient-disjoint training of the detector (M1) and of the lateralization network (M2), the full recording-level
pipeline, the cross-validated baselines, the label-sensitivity analyses and the external check on Siena.

**Language note.** Script docstrings and printed reports are written in French (the authors' working language); the file names,
this README and the application are in English.

## Data (not included)
- TUSZ v2.0.3 (Temple University Hospital EEG Seizure Corpus), distributed by its creators on request:
  https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ - folders `train/`, `dev/`, `eval/`.
- Siena Scalp EEG Database (PhysioNet, open access) - only for the external check (step 15).

Every script takes its paths from the command line (`--help`); the defaults point to a mounted Google Drive layout (`.../projet_propre/`).
Each script prints the block to copy between `<<<COPY_FROM_HERE>>>` and `<<<COPY_TO_HERE>>>` and writes its files in its own folder.
Every script has a `--selftest` that needs no data (and, for most, no GPU).

## Protocol in one paragraph
Patients of `train` and `dev` are pooled and split by patient (step 2a); `eval` is **never** read before a decision rule is frozen on
the validation patients. Preprocessing: 18 bipolar derivations (two mirror chains of 8 + FZ-CZ, CZ-PZ), 250 Hz, 0.5-70 Hz band-pass, 60 Hz
notch, 4 s windows every 2 s, quality control (a recording needs >= 16 valid derivations). Left/right/bilateral labels come from an
algorithmic rule on channel-level annotations (step 1) and were **not reviewed by a clinician**. Scripts that touch `eval` require
`--final_eval` and refuse to run without a frozen rule; they are listed below with the number of passes.

## Layout

| Folder | Content |
|---|---|
| `src/` | All pipeline steps (`step1_...` to `step27_...`) and the network definition `eeg_model.py` |
| `app/` | Inference pipeline (`neurolat_app.py`), Flask API (`server.py`) and non-regression test |
| `frontend/` | Web interface (Next.js) |
| `weights/` | Trained M1 (3 seeds), M2 (3 seeds) and the frozen decision rule `params.json` |

## Steps (what each script does and what it writes)

| Step | Script (`src/`) | Role | Uses `eval`? |
|---|---|---|---|
| 1 | `step1_inventaire_labels.py` | Inventory of annotations; label rule (onset window 2 s, k = 2 channels per side, ratio 2, ties -> bilateral) | no |
| 2a / 2b | `step2a_split_patients.py`, `step2b_cache_windows.py` | Patient split; preprocessing and window cache | no |
| 3 / 3b | `step3_train_m1.py`, `step3b_baselines_m1.py` | M1 training (3 seeds) and simple baselines | no |
| 4 / 4b / 4c | `step4_train_m2.py`, `step4b_baselines_m2.py`, `step4c_compare_m2.py` | M2 training (3 seeds), logistic baselines, paired comparisons with uncertainty | no |
| 5a / 5b | `step5a_score_recordings.py`, `step5b_pipeline_eval.py` | Whole-recording scores; decision rule tuned on validation (`tune`), applied once to `eval` (`final`) | `final` only |
| 6 | `step6_lr_comparator.py` | Logistic-regression comparator for M2 under the same frozen rule | yes (`--final_eval`) |
| 7 | `step7_features_m2_cv.py` | Asymmetry features, patient-level 5-fold cross-validation (B_asym80 etc.) | no |
| 8, 9, 10 | `step8_hard_negatives.py`, `step9_stage2_filter.py`, `step10_gb_detector.py` | Alternative detectors (hard negatives, second stage, gradient-boosted trees, ensemble) | step 10 yes (`--final_eval`) |
| 11-14 | `step11_...` to `step14_label_v2.py` | Label clarity, baseline ratio, M2 ensemble, second label definition | no |
| 15 | `step15_siena_external.py` | External check on Siena (one pass, nothing tuned) | no (Siena) |
| 16-24 | `step16_...` to `step24_foundation_probe.py` | Exploratory attempts on the bilateral class (cross-validation only), incl. pretrained CBraMod probe | no |
| 25 | `step25_label_sensitivity.py` | Sensitivity of labels to the rule (annotations only) | annotations only |
| 26 | `step26_patient_identity.py` | Patient-identity control (annotations only) | annotations only |
| 27 | `step27_rescoring_confusion.py` | Out-of-fold confusion matrix; frozen end-to-end predictions re-scored under each alternative label | re-reads frozen `eval` scores |
| 21 | `step21_audit_pack.py` | Blind clinical review pack of 60 events (not yet reviewed) | no |

The paper-section / figure / table mapping is given in `NUMBERS.md` (generated, see below).

## Typical run order
1. `python src/step1_inventaire_labels.py --selftest` ... (all self-tests need no data)
2. Steps 1, 2a, 2b, 3, 4, 5a (`--role val`), 5b `tune` on validation.
3. Steps 7, 11-14, 16-20, 22-24 (cross-validation, no `eval`).
4. Freeze the rule, then 5a (`--role eval --final_eval`), 5b `final`, 6, 10 (and 15 for Siena).

## Application (web interface)
Two programs, both local:

    pip install -r requirements.txt
    python app/server.py --weights weights/         # Flask API on http://localhost:5000  (POST /analyser, GET /status)
    cd frontend && npm install && npm run dev       # interface on http://localhost:3000

Drop a TUSZ-format `.edf` file on the *Analyse* page. The server applies exactly the evaluated pipeline (`app/neurolat_app.py`: same
preprocessing as step 5a, same frozen rule read from `weights/params.json`, M1 and M2 = mean over 3 seeds) and returns the detections,
the side predicted for each detection and the probability curve. The file is written to a temporary file and deleted after the analysis.
Command line: `python app/neurolat_app.py --edf recording.edf --weights weights/ --out results/` (CSV, PNG timeline, text report).
`python app/test_nonregression.py ...` checks that the application reproduces the stored step-5a scores and the step-5b detections.
The performance figures displayed by the interface are in one file, `frontend/lib/performance.ts`, copied from the step-5b `final` report.
**Research prototype, not a medical device.**

## Reproducibility notes
- Seeds are set; GPU non-determinism can change the last decimals. Results in the paper are means over 3 seeds.
- `eval` was scored for several detectors (M1 rule frozen on validation, logistic comparator, gradient-boosted trees and their ensemble);
  all are reported in the paper. No rule was modified after looking at `eval`.
- The label rule is algorithmic and unreviewed by a clinician; absolute lateralization accuracy is low and should be read against the
  majority-class rate (see step 27).
