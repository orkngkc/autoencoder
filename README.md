# Autoencoder-based Anomaly Detection for Maritime Surveillance

This project presents a custom convolutional autoencoder used for unsupervised anomaly detection in optical remote sensing imagery. The autoencoder is trained exclusively on background-only maritime scenes, allowing it to learn the “normal” patterns of sea environments. Ships and other man-made objects are then detected as structural anomalies during inference based on reconstruction errors.

## 🌊 Context

In maritime monitoring tasks, detecting the presence of potentially dangerous or unknown vessels is critical. However, acquiring labeled ship data can be expensive and incomplete. This project introduces an unsupervised strategy to infer the presence of ships using an autoencoder trained only on ship-free images. During inference, regions with high reconstruction error are flagged as anomalies.

## 📁 Dataset

- Dataset: MASATI (Maritime Satellite Imagery)
- Used a subset of background-only images extracted manually from the MASATI training set
- Image size: All input images were resized to 256×256 for training

## 🧠 Autoencoder Architecture & Training

- Framework: PyTorch 2.0 with CUDA acceleration (A100-SXM4-40GB)
- Input: 256×256 RGB image patches
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 1e−5
- Epochs: 30
- Final training loss: 0.008609

The autoencoder was trained to reconstruct background-only scenes. Ships, which were never seen during training, appear as anomalies in the reconstruction due to their deviation from the learned patterns.

## 🧪 Anomaly Map Computation

After training, the autoencoder processes images that may or may not contain ships. Two different anomaly map strategies were tested:

### 1. Absolute RGB Differences (Aggressive)
This method computes the L1 norm of the difference between input and reconstructed image:
```
Error(i, j) = (|R_input - R_recon| + |G_input - G_recon| + |B_input - B_recon|) / 3
```
- Very sensitive to small differences
- Tends to highlight both real anomalies (ships) and natural variations (coastline textures)

### 2. Signed RGB Differences (Balanced)
This method computes the signed average of RGB differences:
```
Error(i, j) = (R_input - R_recon + G_input - G_recon + B_input - B_recon) / 3
```
- More balanced
- Better at suppressing minor variations and highlighting true structural anomalies (e.g., ships in open sea)

## 📊 Results & Insights

- Both methods successfully identified ship regions as anomalies
- Absolute RGB method often resulted in high false positives (shoreline falsely detected)
- Signed RGB method provided more reliable and localized anomaly maps
- The model performed well on open-sea images with ships
- False positives increased near ports or coastlines due to the presence of buildings, piers, or vehicles

## ⚠️ Limitations

- Not a conventional object detector — it does not output bounding boxes or classifications
- May confuse complex man-made shoreline features with ships
- Attempts to use deeper encoders (e.g., VGG19) showed no meaningful improvement due to hardware limitations

## 💡 Future Work

- Integrate anomaly detection maps with object detectors
- Use more diverse background training data including ports
- Re-implement with deeper architectures using more powerful GPUs

## 👥 Authors

- Melih Kaan Şahinbaş: Autoencoder research, anomaly detection experiments
- Orkun Efe Gökçe: Model training, unsupervised inference, visualizations

## 📎 Reference

- MASATI Dataset: [https://www.sciencedirect.com/science/article/pii/S235248552400608X](https://www.sciencedirect.com/science/article/pii/S235248552400608X)

## 📄 License

This project is for academic and research purposes only.
