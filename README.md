# MDenseU-Net: Multi-Scale Dense Segmentation with SE Attention

> A novel medical image segmentation architecture combining dense connectivity,
> multi-scale feature extraction, and Squeeze-and-Excitation channel attention
> within a U-Net encoder-decoder framework. Evaluated on lung CT segmentation,
> MDenseU-Net outperforms standard U-Net on **14/16 metrics** while using **52% fewer parameters**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Qualitative Results](#qualitative-results)
- [Grad-CAM Visualizations](#grad-cam-visualizations)
- [Learning Curves](#learning-curves)
- [Metrics Table](#metrics-table)
- [Setup](#setup)
- [Relevance to Medical AI Research](#relevance-to-medical-ai-research)

---

## Overview

This project proposes **MDenseU-Net**, a novel 2D medical image segmentation architecture
benchmarked against standard **U-Net** on the
[Finding Lungs in CT Data](https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data) dataset.

### Key Contributions

- **Dense encoder blocks** with multi-scale parallel convolutions (3×3, 5×5, dilated 3×3)
  capturing both fine-grained boundary details and global contextual features
- **Squeeze-and-Excitation (SE) attention** after each dense block for channel-wise
  recalibration of feature responses
- **Bilinear upsampling decoder** with U-Net style skip connections to avoid
  checkerboard artifacts common in transposed convolution decoders
- **Deep supervision** with auxiliary segmentation outputs at intermediate decoder
  stages to improve gradient flow during training
- **Comprehensive evaluation** across 16 metrics covering overlap, boundary,
  statistical, and calibration dimensions

---

## Architecture

![MDenseU-Net Architecture](assets/architecture.png)

### MDenseU-Net Design

Each encoder stage consists of a `DenseMultiScaleBlock`:
- Three stacked `MultiScaleDenseLayer` units, each receiving concatenated
  feature maps from all preceding layers (dense connectivity)
- Each layer applies three parallel convolutions: 3×3, 5×5, and dilated 3×3 (d=2)
  whose outputs are concatenated to form a rich multi-scale representation
- A `SEBlock` at the end of each dense block recalibrates channel importance
  via global average pooling followed by two fully-connected layers and sigmoid gating

The decoder uses `UpBlockBilinear` modules — bilinear interpolation followed
by concatenation with the corresponding encoder skip connection and double conv layers.

Deep supervision adds auxiliary 1×1 convolution outputs at decoder stages D2 and D3,
combined in the training loss with weights `[0.5 main, 0.25 aux3, 0.25 aux2]`.

### Parameter Comparison

| Model | Parameters | Reduction |
|---|---|---|
| U-Net | 7,762,465 | — |
| MDenseU-Net | 3,745,953 | **−52%** |

---

## Dataset

**Finding and Measuring Lungs in CT Data** (Kaggle — kmader)

- 267 paired 2D CT slices with binary lung segmentation masks
- Format: `.tif` images in `2d_images/` and `2d_masks/` folders
- Preprocessing: intensity normalization to `[0, 1]`, resize to `256×256`
- Augmentation: horizontal flip, rotation ±10°, elastic deformation,
  brightness/contrast jitter
- Split: 70% train / 15% validation / 15% test

**Class distribution:** ~21.5% foreground (lung) / ~78.5% background —
handled via `BCEWithLogitsLoss` with `pos_weight=3.0`

---

## Training

Both models trained under identical conditions for fair comparison:

| Setting | Value |
|---|---|
| Loss | BCE + Dice (α=0.5 each) |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-5 |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Batch size | 8 |
| Max epochs | 50 |
| Early stopping | Patience = 10 on val Dice |
| Mixed precision | torch.cuda.amp (AMP) |
| Hardware | Google Colab T4 GPU |
| Random seed | 42 |

---

## Results

MDenseU-Net achieves superior performance across nearly all metrics
with significantly fewer parameters.

### Summary

| Model | Dice | IoU | HD95 | ECE | Brier | Params |
|---|---|---|---|---|---|---|
| U-Net | 0.9744 | 0.9504 | 5.917 | 0.157 | 0.0366 | 7.76M |
| **MDenseU-Net** | **0.9806** | **0.9624** | **5.196** | **0.111** | **0.0217** | **3.75M** |
| Improvement | +0.64% | +1.25% | −12.19% | −29.22% | −40.60% | **−52%** |

---

## Qualitative Results

Predicted segmentation overlays on test samples.
Colors: **green** = True Positive, **red** = False Positive, **blue** = False Negative.

![Segmentation Results](assets/sample_segmentation.png)

MDenseU-Net produces cleaner boundaries with less false positive noise
around lung edges compared to U-Net, consistent with the lower HD95 and ASSD scores.

---

## Grad-CAM Visualizations

Gradient-weighted Class Activation Maps showing where each model focuses attention.

![Grad-CAM](assets/gradcam.png)

MDenseU-Net shows more fine-grained internal structure attention (vessels, airways)
compared to U-Net's smoother, more uniform activation — reflecting the multi-scale
convolutions capturing richer anatomical detail.

---

## Learning Curves

Training and validation loss and Dice score over epochs for both models.

![Learning Curves](assets/learning_curves.png)

MDenseU-Net converges faster and achieves lower training loss (0.1475 vs 0.2483)
while maintaining better generalization on the validation set.

---

## Metrics Table

Full evaluation on the held-out test set (16 metrics):

| # | Metric | U-Net | MDenseU-Net | Delta | % Improvement |
|---|---|---|---|---|---|
| 1 | Dice | 0.97445 | 0.98065 | +0.00620 | +0.64% |
| 2 | IoU | 0.95045 | 0.96235 | +0.01191 | +1.25% |
| 3 | VOE ↓ | 0.04955 | 0.03765 | −0.01191 | −24.02% |
| 4 | HD ↓ | 15.100 | 14.063 | −1.037 | −6.87% |
| 5 | HD95 ↓ | 5.917 | 5.196 | −0.721 | −12.19% |
| 6 | ASSD ↓ | 1.085 | 0.888 | −0.197 | −18.15% |
| 7 | MSD ↓ | 1.085 | 0.888 | −0.197 | −18.15% |
| 8 | Precision | 0.95448 | 0.96810 | +0.01363 | +1.43% |
| 9 | Recall | 0.99564 | 0.99391 | −0.00173 | −0.17% |
| 10 | Specificity | 0.98657 | 0.99076 | +0.00419 | +0.42% |
| 11 | F1 | 0.97445 | 0.98065 | +0.00620 | +0.64% |
| 12 | MCC | 0.96750 | 0.97537 | +0.00788 | +0.81% |
| 13 | Kappa | 0.96702 | 0.97510 | +0.00808 | +0.84% |
| 14 | Balanced Acc | 0.99111 | 0.99233 | +0.00123 | +0.12% |
| 15 | ECE ↓ | 0.15747 | 0.11146 | −0.04601 | −29.22% |
| 16 | Brier ↓ | 0.03657 | 0.02173 | −0.01485 | −40.60% |

> ↓ = lower is better. MDenseU-Net wins on **14 out of 16 metrics**.

---

## Setup

### Requirements

```bash
pip install torch torchvision monai nibabel SimpleITK kagglehub
pip install scikit-learn scipy matplotlib seaborn pandas torchsummary
```

### Run on Google Colab

The entire project is contained in a single notebook designed to run
sequentially on Google Colab free tier (T4 GPU):

```
MDenseUNet_Colab_Research_Notebook.ipynb
```

Open in Colab → Runtime → Run All

The notebook will automatically:
1. Download the dataset via `kagglehub`
2. Preprocess and split the data
3. Train both U-Net and MDenseU-Net
4. Evaluate all 16 metrics
5. Generate all visualizations

---

## Relevance to Medical AI Research

This project directly addresses core challenges in medical image analysis:

**Segmentation as a foundation for synthesis** — accurate segmentation masks
are a prerequisite for conditional image generation tasks such as pseudo-CT
synthesis from MRI, which is central to digital twin frameworks for radiotherapy planning.

**Multi-modal medical imaging** — the multi-scale dense architecture is designed
to generalize across imaging modalities (CT, MRI, PET), making it transferable
to multimodal segmentation pipelines.

**Uncertainty and calibration** — the 40% improvement in Brier score and 29%
improvement in ECE demonstrate that MDenseU-Net produces well-calibrated
probability outputs, which is critical for clinical deployment where confidence
estimates must be reliable.

**Parameter efficiency** — achieving superior performance with 52% fewer
parameters demonstrates that architectural design choices (dense connectivity,
multi-scale convolutions, SE attention) are more impactful than raw model capacity.

---

## Stack

Python · PyTorch · MONAI · scikit-learn · scipy · matplotlib · seaborn · pandas · Google Colab

---

## Author

**Ayoub EL KHAIARI**
MSc Advanced Machine Learning and Multimedia Intelligence — USMBA, Fez
[GitHub](https://github.com/Ayoub-Elkhaiari) · [Portfolio](https://ayoub-elkhaiari.netlify.app/)