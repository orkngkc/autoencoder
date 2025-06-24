# autoencoder

# Autoencoder-based Anomaly Detection for Maritime Ship Monitoring

This project introduces a custom convolutional autoencoder for **unsupervised anomaly detection** in optical remote sensing imagery, focusing on **maritime surveillance**. The goal is to detect ships as **structural anomalies** against a learned “normal” background (open sea, coastal areas, etc.), without requiring any labeled ship data.

---

## 🌊 Overview

Traditional ship detectors need labeled data and often struggle under variable maritime conditions. This project explores an **unsupervised approach**: train an autoencoder on **background-only images** (no ships) so that any reconstruction error in test images indicates the **presence of anomalous content**, i.e., ships.

---

## 📁 Dataset

- **Source**: MASATI dataset
- **Usage**: Background-only images (no ships) were manually extracted from the MASATI training set.
- **Preprocessing**: All images resized to **256×256**.

---

## 🧠 Autoencoder Architecture & Training

| Component        | Details |
|------------------|---------|
| Framework        | PyTorch 2.0 (CUDA-enabled) |
| Input            | 256×256 RGB image patches |
| Loss Function    | Mean Squared Error (MSE) |
| Optimizer        | Adam |
| Learning Rate    | 0.001 |
| Weight Decay     | 1e-5 |
| Epochs           | 30 |
| Final Training Loss | **0.008609** |
| Hardware         | NVIDIA A100-SXM4-40GB (Google Colab Pro) |

---

## 🧪 Inference Strategy

After training, the autoencoder was evaluated on both:
- **Ship-present** images (anomalous)
- **Background-only** images (normal)

The key idea is that the **reconstruction error** should spike in regions containing ships.

---

## 📊 Anomaly Map Computation

Two strategies were implemented:

### 1. Absolute RGB Differences (Aggressive)
```
Error(i, j) = (|Rin - Rrec| + |Gin - Grec| + |Bin - Brec|) / 3
Pros: Highly sensitive to local changes

Cons: Often falsely flags coastal textures as anomalies
