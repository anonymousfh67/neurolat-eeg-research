# -*- coding: utf-8 -*-
"""
NeuroLat — Backend Flask pour la latéralisation EEG
M1 : model_m1_v2.pt  |  M2 : model_m2_v3.pt
Pipeline v3 : seuil M1 adaptatif (percentile 80) + règle de décision par ratio
Adapté pour tourner en LOCAL (pas Colab) et pour être appelé par le
frontend React/Next.js généré avec v0 (CORS activé).
"""

import os, tempfile
import numpy as np
import torch, torch.nn as nn
import mne
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from collections import Counter

mne.set_log_level("WARNING")

# ════════════════════════════════════════════════════
# CONFIGURATION — chemin LOCAL (pas Colab/Drive)
# ════════════════════════════════════════════════════
MODELS     = "./models"
FS         = 250
WIN_SAMP   = 1000
N_CANAUX   = 19
SEUIL_M1   = 0.35
CLASSES_M2 = ["gauche", "droite", "bilateral"]

# Seuil de ratio pour départager gauche/droite plutôt que de tomber
# systématiquement sur "bilateral" dès que les deux sont présents.
# ratio = max(pct_g, pct_d) / min(pct_g, pct_d)
# Avec RATIO_SEUIL = 1.4 : un écart de 45.6% vs 38.2% (ratio ≈ 1.19)
# reste sous le seuil -> NE PAS forcer bilateral dans ce cas précis,
# donc on veut au contraire un seuil plus élevé pour que ce cas devienne
# "gauche" (le vote réellement majoritaire) plutôt que "bilateral".
# On l'augmente à 1.5 : seuls des votes vraiment proches (ratio < 1.5,
# soit par ex. 45% vs 40%) restent classés bilatéral ; un écart plus
# net (comme 45.6% vs 38.2%, ratio ≈ 1.19... non, regardons bien :
# en réalité ratio=1.19 < 1.5 -> resterait bilateral). Pour que VOTRE
# cas (45.6 vs 38.2) bascule en "gauche", il faut un seuil plus BAS,
# pas plus haut : on veut que ratio=1.19 soit considéré comme "assez
# différent" pour ne plus être bilatéral. D'où RATIO_SEUIL = 1.15.
RATIO_SEUIL = 1.15

PAIRES = [
    ("FP1","F7"), ("F7","T3"),
    ("T3","T5"),  ("T5","O1"),
    ("FP1","F3"), ("F3","C3"),
    ("C3","P3"),  ("P3","O1"),
    ("FP2","F8"), ("F8","T4"),
    ("T4","T6"),  ("T6","O2"),
    ("FP2","F4"), ("F4","C4"),
    ("C4","P4"),  ("P4","O2"),
    ("FZ","CZ"),  ("CZ","PZ"),
    ("CZ","C4"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════
# ARCHITECTURE
# ════════════════════════════════════════════════════
class AttentionSpatiale(nn.Module):
    def __init__(self, n_canaux=19):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_canaux, n_canaux*2), nn.ReLU(),
            nn.Linear(n_canaux*2, n_canaux), nn.Softmax(dim=-1))
    def forward(self, x):
        return x * self.attention(x.mean(dim=-1)).unsqueeze(-1)


class PatchEmbedding(nn.Module):
    def __init__(self, n_canaux=19, d_model=64):
        super().__init__()
        self.conv_temp = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1,25), padding=(0,12)),
            nn.BatchNorm2d(40), nn.ELU(), nn.Dropout(0.3))
        self.conv_spat = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=(n_canaux,1)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)), nn.Dropout(0.3))
        self.proj = nn.Linear(40, d_model)
    def forward(self, x):
        b, c, t = x.shape
        x = self.conv_temp(x.unsqueeze(1))
        x = self.conv_spat(x)
        return self.proj(x.squeeze(2).permute(0,2,1))


class ConformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, ff_mult=4, kernel=31, dropout=0.3):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model*ff_mult),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*ff_mult, d_model), nn.Dropout(dropout))
        self.norm_attn = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model*2, 1), nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=kernel,
                      padding=kernel//2, groups=d_model),
            nn.BatchNorm1d(d_model), nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout))
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model*ff_mult),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*ff_mult, d_model), nn.Dropout(dropout))
        self.norm_final = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + 0.5*self.ff1(x)
        xn = self.norm_attn(x)
        ao, _ = self.attention(xn, xn, xn)
        x = x + self.drop_attn(ao)
        xn = self.norm_conv(x)
        xc = self.conv(xn.permute(0,2,1))
        x = x + xc.permute(0,2,1)
        x = x + 0.5*self.ff2(x)
        return self.norm_final(x)


class EEGConformer(nn.Module):
    def __init__(self, n_canaux=19, d_model=64, n_heads=4,
                 n_blocks=4, n_classes=2, dropout=0.3):
        super().__init__()
        self.attn_spatiale = AttentionSpatiale(n_canaux)
        self.patch_embed   = PatchEmbedding(n_canaux, d_model)
        self.pos_drop      = nn.Dropout(dropout)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_blocks)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model//2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model//2, n_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, x):
        x = self.attn_spatiale(x)
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for block in self.conformer_blocks: x = block(x)
        return self.classifier(x.mean(dim=1))


# ════════════════════════════════════════════════════
# CHARGEMENT MODÈLES
# ════════════════════════════════════════════════════
model_m1 = None
model_m2 = None
mu1 = std1 = mu2 = std2 = None

def charger_modeles():
    global model_m1, model_m2, mu1, std1, mu2, std2
    try:
        model_m1 = EEGConformer(n_classes=2).to(device)
        model_m1.load_state_dict(torch.load(
            f"{MODELS}/model_m1_v2.pt", map_location=device))
        model_m1.eval()
        for p in model_m1.parameters(): p.requires_grad_(False)

        model_m2 = EEGConformer(n_classes=3).to(device)
        model_m2.load_state_dict(torch.load(
            f"{MODELS}/model_m2_v3.pt", map_location=device))
        model_m2.eval()
        for p in model_m2.parameters(): p.requires_grad_(False)

        mu1  = float(np.load(f"{MODELS}/mu1_v2.npy")[0])
        std1 = float(np.load(f"{MODELS}/std1_v2.npy")[0])
        mu2  = float(np.load(f"{MODELS}/mu2_v3.npy")[0])
        std2 = float(np.load(f"{MODELS}/std2_v3.npy")[0])
        print(f"✅ Modèles chargés sur {device}")
        return True
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False


# ════════════════════════════════════════════════════
# PRÉTRAITEMENT EDF
# ════════════════════════════════════════════════════
def nettoyer_nom(ch):
    ch = ch.upper().strip()
    for pref in ["EEG ", "EEG"]:
        if ch.startswith(pref): ch = ch[len(pref):]
    for suf in ["-REF","-LE","-AR","-AVG","-A1","-A2"]:
        if ch.endswith(suf): ch = ch[:-len(suf)]
    return ch.strip()


def pretraiter_edf_v3(edf_path):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        exclus = ["EKG","EMG","PG","SP","ECG","EOG","BURST",
                  "SUPPR","RLC","LUC","OZ","PG1","PG2","T1",
                  "T2","30","31","32","PHOTIC","DC"]
        ch_eeg = [c for c in raw.ch_names
                  if "EEG" in c.upper()
                  and not any(x in c.upper() for x in exclus)]
        if len(ch_eeg) < 5: return None
        raw = raw.pick(ch_eeg)
        raw = raw.filter(0.5, 70.0, verbose=False)
        raw = raw.notch_filter(60, verbose=False)
        if sfreq != FS: raw = raw.resample(FS, verbose=False)
        data_raw = raw.get_data() * 1e6
        ch_dict = {}
        for i, ch in enumerate(raw.ch_names):
            nom = nettoyer_nom(ch)
            if 1.0 < data_raw[i].std() < 500.0:
                ch_dict[nom] = data_raw[i]
        if len(ch_dict) < 5: return None
        n_pts = data_raw.shape[1]
        data_m = np.zeros((N_CANAUX, n_pts), dtype=np.float32)
        for i, (a, b) in enumerate(PAIRES[:N_CANAUX]):
            if a in ch_dict and b in ch_dict:
                data_m[i] = ch_dict[a] - ch_dict[b]
            elif a in ch_dict:
                data_m[i] = ch_dict[a]
        non_nuls = sum(1 for i in range(N_CANAUX) if data_m[i].std() > 0.5)
        if non_nuls < 8: return None
        return data_m
    except Exception:
        return None


def normaliser_segment(seg):
    mu  = seg.mean(axis=1, keepdims=True)
    std = seg.std(axis=1, keepdims=True) + 1e-8
    return np.clip((seg-mu)/std, -10, 10).astype(np.float32)


# ════════════════════════════════════════════════════
# PIPELINE DE PRÉDICTION
# ════════════════════════════════════════════════════
def predire_pipeline(data):
    if model_m1 is None or model_m2 is None:
        return None, "Modèles non chargés"

    segs = []
    pos = 0
    step = max(1, WIN_SAMP // 2)
    while pos + WIN_SAMP <= data.shape[1]:
        segs.append(normaliser_segment(data[:, pos:pos+WIN_SAMP].copy()))
        pos += step
    if not segs:
        return None, "Signal trop court"

    X = np.array(segs)
    X_m1 = (X - mu1) / std1

    # ÉTAPE 1 — Scorer toutes les fenêtres avec M1
    all_probs_m1 = []
    for i in range(0, len(X_m1), 64):
        Xb = torch.tensor(X_m1[i:i+64]).float().to(device)
        with torch.no_grad():
            p1 = torch.softmax(model_m1(Xb), dim=1)
            for prob in p1[:, 1].cpu().numpy():
                all_probs_m1.append(float(prob))

    probs_arr    = np.array(all_probs_m1)
    masque_crise = probs_arr >= SEUIL_M1
    n_detectees  = masque_crise.sum()

    vide = {
        "prediction": "non_detecte",
        "label": "Aucune crise détectée",
        "votes": {"gauche": 0, "droite": 0, "bilateral": 0},
        "probs": [0.0, 0.0, 0.0],
        "n_fenetres": len(segs),
        "n_crises": 0,
        "pct_crises": 0.0
    }
    if n_detectees == 0:
        return vide, None

    # ÉTAPE 2 — Seuil adaptatif (percentile 80 des fenêtres détectées)
    seuil_adaptatif = max(
        0.50,
        float(np.percentile(probs_arr[masque_crise], 80)))

    # ÉTAPE 3 — M2 sur les meilleures fenêtres (seuil adaptatif)
    preds_m2, probs_all = [], []
    for i in range(0, len(X_m1), 64):
        Xb = torch.tensor(X_m1[i:i+64]).float().to(device)
        with torch.no_grad():
            p1 = torch.softmax(model_m1(Xb), dim=1)
            est_crise = (p1[:, 1] >= seuil_adaptatif).cpu().numpy()
        for j, is_c in enumerate(est_crise):
            if is_c:
                seg = (X[i+j:i+j+1] - mu2) / std2
                Xt2 = torch.tensor(seg).float().to(device)
                with torch.no_grad():
                    p2 = torch.softmax(model_m2(Xt2), dim=1)
                    probs = p2[0].cpu().numpy()
                    probs_all.append(probs)
                    preds_m2.append(CLASSES_M2[probs.argmax()])

    # Fallback au seuil normal si rien n'est détecté au seuil adaptatif
    if not preds_m2:
        for i in range(0, len(X_m1), 64):
            Xb = torch.tensor(X_m1[i:i+64]).float().to(device)
            with torch.no_grad():
                p1 = torch.softmax(model_m1(Xb), dim=1)
                est_crise = (p1[:, 1] >= SEUIL_M1).cpu().numpy()
            for j, is_c in enumerate(est_crise):
                if is_c:
                    seg = (X[i+j:i+j+1] - mu2) / std2
                    Xt2 = torch.tensor(seg).float().to(device)
                    with torch.no_grad():
                        p2 = torch.softmax(model_m2(Xt2), dim=1)
                        probs = p2[0].cpu().numpy()
                        probs_all.append(probs)
                        preds_m2.append(CLASSES_M2[probs.argmax()])

    if not preds_m2:
        return vide, None

    ct    = Counter(preds_m2)
    total = sum(ct.values())
    votes = {
        "gauche":    int(ct.get("gauche", 0)),
        "droite":    int(ct.get("droite", 0)),
        "bilateral": int(ct.get("bilateral", 0))
    }

    probs_moy = np.mean(probs_all, axis=0)
    p_b   = float(probs_moy[2])
    pct_g = votes["gauche"] / total
    pct_d = votes["droite"] / total

    # Règle finale — départage gauche/droite par ratio plutôt que de
    # forcer "bilateral" dès que les deux sont simplement présents.
    if p_b > 0.40:
        prediction = "bilateral"
    elif pct_g > 0.20 and pct_d > 0.20:
        ratio = max(pct_g, pct_d) / (min(pct_g, pct_d) + 1e-8)
        if ratio < RATIO_SEUIL:
            prediction = "bilateral"
        elif pct_g > pct_d:
            prediction = "gauche"
        else:
            prediction = "droite"
    elif pct_g > 0.50:
        prediction = "gauche"
    elif pct_d > 0.50:
        prediction = "droite"
    else:
        prediction = CLASSES_M2[int(probs_moy.argmax())]

    labels = {
        "gauche":    "Foyer Gauche",
        "droite":    "Foyer Droit",
        "bilateral": "Bilatéral"
    }
    return {
        "prediction": prediction,
        "label":      labels[prediction],
        "votes":      votes,
        "probs":      probs_moy.tolist(),
        "n_fenetres": len(segs),
        "n_crises":   total,
        "pct_crises": round(total / len(segs) * 100, 1)
    }, None


# ════════════════════════════════════════════════════
# ROUTES FLASK — CORS activé pour appel depuis localhost:3000
# ════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


@app.route("/")
def index():
    return jsonify({
        "status": "NeuroLat backend actif",
        "info": "Le frontend tourne séparément sur http://localhost:3000"
    })


@app.route("/status")
def status():
    return jsonify({"ok": model_m1 is not None and model_m2 is not None})


@app.route("/analyser", methods=["POST"])
def analyser():
    if model_m1 is None or model_m2 is None:
        return jsonify({"error": "Modèles non chargés."})
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu."})
    f = request.files["file"]
    if not f.filename.lower().endswith(".edf"):
        return jsonify({"error": "Utilisez un fichier .edf"})

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        data = pretraiter_edf_v3(tmp_path)
        if data is None:
            return jsonify({"error": "Impossible de lire le fichier EDF. Vérifiez le format et les canaux."})
        result, err = predire_pipeline(data)
        if err:
            return jsonify({"error": err})
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erreur serveur : {str(e)}"})
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    charger_modeles()
    app.run(host="0.0.0.0", port=5000, debug=False)