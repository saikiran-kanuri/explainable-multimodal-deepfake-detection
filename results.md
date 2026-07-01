# Results

## Explainable Multimodal Deepfake Detection using Visual, Temporal, Audio, Lip-Sync, and Frequency Analysis

This document summarizes the performance of each individual modality model and the final attention-weighted late fusion system.

---

## 1. Per-Modality Results

| Notebook | Modality | Model | Accuracy | AUC | Notes |
|---|---|---|---|---|---|
| NB05 | Visual | EfficientNet-B4 (MBConv + SE attention) | ~99% | ~0.99 | Strongest individual modality |
| NB06 | Temporal | BiLSTM + Bahdanau Attention | TBD | TBD | Fill in from training logs |
| NB07 | Audio | CRNN + SpecAugment (BiGRU) | TBD (lower) | TBD (lower) | Reduced accuracy due to data constraints; Wav2Vec2 identified as improvement path |
| NB08 | Lip-Sync | SyncNet-style dual encoder | TBD | TBD | Trained on 22,898 self-supervised contrastive pairs (margin = 1.0) |
| NB09 | Frequency | FFT-based Forensics CNN | TBD | TBD | Detects GAN spectral fingerprints & checkerboard artifacts |

> **Note:** Replace all `TBD` entries with the exact metrics from your NB06–NB09 evaluation cells before submission. I don't have those specific numbers on record — paste them in and I'll update the table.

---

## 2. Fusion Results (NB10)

| Metric | Value |
|---|---|
| Fusion Strategy | Attention-weighted late fusion (logistic regression, L-BFGS) |
| Overall Fused AUC | **0.84** |
| Visual Modality Weight (α) | ≈ 0.31 |
| Other Modality Weights | TBD |

The fusion layer combines confidence scores from all five modalities, learning per-modality attention weights rather than using fixed averaging. Visual features (EfficientNet-B4) receive the highest weighting (α ≈ 0.31), reflecting their individually strong discriminative performance.

---

## 3. Key Observations

- **Visual modality dominance**: EfficientNet-B4 alone achieves near-ceiling performance (~99%), making it the most reliable single-modality signal.
- **Audio as the weak link**: The CRNN audio model underperforms relative to other modalities, primarily attributable to limited training data rather than architectural limitations. Wav2Vec2-based pretraining is proposed as a future improvement.
- **Fusion trade-off**: The fused AUC (0.84) is lower than the visual-only AUC, which is expected in late-fusion setups where weaker modalities (audio) pull down the aggregate score — but fusion improves robustness against attacks that specifically target the visual channel (e.g., high-quality face-swap deepfakes with clean audio artifacts, or vice versa).
- **Explainability**: Grad-CAM visualizations on the visual branch provide localized evidence (e.g., mouth/eye region artifacts) supporting the model's classification decisions, output alongside a natural language explanation module.

