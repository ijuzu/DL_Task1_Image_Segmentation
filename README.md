# DL_Task1_Image_Segmentation

# Image Segmentation for Cat, Dog, and Car Classes

## Overview

This project implements an image segmentation model which classifies every pixel in an image into Background, Cat, Dog, and Car Classes:

| Class | Index | Colour in output |
|-------|-------|-----------------|
| Background | 0 | Black |
| Cat | 1 | Red |
| Dog | 2 | Green |
| Car | 3 | Blue |

The model is a U-Net architecture, trained and evaluated on images from the [Open Images v7] (https://storage.googleapis.com/openimages/web/index.html) dataset.

---

## Architecture

### U-Net

```
Input [B, 3, 256, 256]
        ↓
ENCODER
Block 1: Conv(3→64)×2 + BN + ReLU + SE    → skip1 [B,64,256,256]  MaxPool
Block 2: Conv(64→128)×2 + BN + ReLU + SE  → skip2 [B,128,128,128] MaxPool
Block 3: Conv(128→256)×2 + BN + ReLU + SE → skip3 [B,256,64,64]   MaxPool
Bottleneck: Conv(256→256)×2 + BN + ReLU + SE + Dropout(0.4) [B,256,32,32]
        ↓
DECODER
Up1: ConvTranspose → AttentionGate(skip3) → concat → ConvBlock+SE → [B,128,64,64]
Up2: ConvTranspose → AttentionGate(skip2) → concat → ConvBlock+SE → [B,64,128,128]
Up3: ConvTranspose → AttentionGate(skip1) → concat → ConvBlock+SE → [B,32,256,256]
        ↓
Last layer: Conv2d(32→4, kernel=1×1) → [B, 4, 256, 256]
```

## Loss Function

Combined loss: **0.5 × CrossEntropy + 0.5 × ForegroundDice**

**CrossEntropy** with class weights `[0.1, 5.0, 5.0, 1.5]`:
- Background weight 0.1 
- Cat/Dog weight 5.0: adjusted to be higher to encourage the model to learn small, visually similar classes
- Car weight 1.5: lowered to prevent Car from dominating predictions

**ForegroundDice** (background excluded):
- Dice = 1 − (2×TP) / (2×TP + FP + FN)

---

## Dataset

**Source:** Open Images v7 via FiftyOne  
**Split:**

| Split | Size | Purpose |
|-------|------|---------|
| Train | 300 | Model learns from these images |
| Val | 75 | Monitors training for early stopping |
| Test | 100 | The final metrics are reported for these images |

---

## Training

- Optimizer: Adam, lr=1e-3, weight_decay=1e-4  
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=4)  
- Early stopping: patience=10 
- Max epochs: 20 
- Batch size: 4 (train), 2 (val), 1 (test)  

---

## Metrics

After evaluation on 100 unseen test images, pixel accuracy, in addition to macro and per-class precision, recall, and F1 are computed. 

---

## Benchmark Comparison

The model is benchmarked against **FCN-ResNet50 pretrained on PASCAL VOC** evaluated zero-shot. PASCAL VOC class predictions are mapped to the classes in this model in this way: cat(8)→Cat, dog(12)→Dog, car(7)→Car.

---

## Residual Analysis

The residual analysis identifies the 5 images where the model performs worst and prints per-class false positive and false negative rates across all 100 test images.
