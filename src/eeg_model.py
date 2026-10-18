# -*- coding: utf-8 -*-
"""
eeg_model.py -- EEGConformer (même architecture que dans les travaux précédents), 18 canaux d'entrée.
Partagé par M1 (2 classes) et M2 (3 classes). Nécessite torch.
Remarque : avec une normalisation par canal et par fenêtre, la moyenne temporelle de chaque canal est ~0, donc le module
« AttentionSpatiale » reçoit une entrée presque constante : il agit comme un gain de canal appris, presque indépendant du signal.
"""
import torch
import torch.nn as nn


class AttentionSpatiale(nn.Module):
    def __init__(self, n_canaux=18):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(n_canaux, n_canaux * 2), nn.ReLU(),
                                       nn.Linear(n_canaux * 2, n_canaux), nn.Softmax(dim=-1))

    def forward(self, x):
        return x * self.attention(x.mean(dim=-1)).unsqueeze(-1)


class PatchEmbedding(nn.Module):
    def __init__(self, n_canaux=18, d_model=64):
        super().__init__()
        self.conv_temp = nn.Sequential(nn.Conv2d(1, 40, (1, 25), padding=(0, 12)), nn.BatchNorm2d(40), nn.ELU(), nn.Dropout(0.3))
        self.conv_spat = nn.Sequential(nn.Conv2d(40, 40, (n_canaux, 1)), nn.BatchNorm2d(40), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.3))
        self.proj = nn.Linear(40, d_model)

    def forward(self, x):
        x = self.conv_temp(x.unsqueeze(1))
        x = self.conv_spat(x)
        return self.proj(x.squeeze(2).permute(0, 2, 1))


class ConformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, ff_mult=4, kernel=31, dropout=0.3):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_model * ff_mult, d_model), nn.Dropout(dropout))
        self.norm_attn = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(nn.Conv1d(d_model, d_model * 2, 1), nn.GLU(dim=1),
                                  nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model),
                                  nn.BatchNorm1d(d_model), nn.SiLU(), nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout))
        self.ff2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_model * ff_mult, d_model), nn.Dropout(dropout))
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        xn = self.norm_attn(x)
        ao, _ = self.attention(xn, xn, xn)
        x = x + self.drop_attn(ao)
        xc = self.conv(self.norm_conv(x).permute(0, 2, 1))
        x = x + xc.permute(0, 2, 1)
        x = x + 0.5 * self.ff2(x)
        return self.norm_final(x)


class EEGConformer(nn.Module):
    def __init__(self, n_canaux=18, d_model=64, n_heads=4, n_blocks=4, n_classes=2, dropout=0.3, use_attn=True):
        super().__init__()
        self.attn_spatiale = AttentionSpatiale(n_canaux) if use_attn else nn.Identity()
        self.patch_embed = PatchEmbedding(n_canaux, d_model)
        self.pos_drop = nn.Dropout(dropout)
        self.conformer_blocks = nn.ModuleList([ConformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_blocks)])
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(d_model // 2, n_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.pos_drop(self.patch_embed(self.attn_spatiale(x)))
        for b in self.conformer_blocks:
            x = b(x)
        return self.classifier(x.mean(dim=1))
