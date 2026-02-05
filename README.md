# Diffusion-Enhanced Latent Representation and Transformer-Based Metric Learning  
## for Cross-Domain Ultrasound Image Retrieval

This repository provides the official implementation of the paper:

**Diffusion-Enhanced Latent Representation and Transformer-Based Metric Learning for Cross-Domain Ultrasound Image Retrieval**

The code implements a content-based image retrieval (CBIR) framework for breast ultrasound images, designed to handle domain shifts across different datasets and acquisition settings.

---

## 🔍 Overview

Breast ultrasound image retrieval is challenging due to:
- strong speckle noise,
- low contrast,
- high intra-class similarity,
- and significant domain variations across scanners and datasets.

To address these challenges, we propose a **hybrid retrieval framework** that combines:

1. **VAE-based latent encoding** for structural representation  
2. **Diffusion-based feature refinement** for noise-robust enhancement  
3. **Metric learning with triplet loss** to optimize retrieval similarity  
4. **Cosine-distance-based retrieval** for stable ranking

The framework is evaluated under both **in-domain** and **cross-domain** settings using publicly available ultrasound datasets.

---

## 🧠 Methodology

### 1. Variational Autoencoder (VAE)
- Input: grayscale ultrasound images
- Output: compact latent representations capturing anatomical structure
- Purpose: reduce noise sensitivity and enforce structured embeddings

### 2. Diffusion-Based Feature Refinement
- Applied in image space to model speckle-noise characteristics
- Enhances robustness to acquisition variability
- Used as a complementary refinement mechanism

### 3. Feature Fusion
- Latent VAE embeddings are combined with diffusion-refined features
- Produces a unified representation for retrieval

### 4. Metric Learning
- Batch-Hard Triplet Loss is used to:
  - minimize intra-class distance
  - maximize inter-class separation
- Embeddings are L2-normalized before retrieval

### 5. Retrieval
- Similarity measured using **cosine distance**
- Evaluation reported using **Precision@5, Recall@5, mAP@5, MAR@5, and AUC**

---
## Retrieval Results

<p align="center">
  <img src="results/retrieval_results.png" width="600"/>
</p>


## 📊 Datasets

The framework is evaluated using the following datasets:

### Training / In-domain
- **BUSI (Breast Ultrasound Images Dataset)**
- **HiSBreast Breast Ultrasound Dataset**

### Cross-domain Testing
- **BUS-UCLM Breast Ultrasound Dataset**


> ⚠️ To avoid information leakage, strict query–gallery splits are applied during evaluation.

---

## ⚙️ Installation

```bash
pip install kagglehub torch torchvision timm albumentations opencv-python scikit-learn faiss-cpu
