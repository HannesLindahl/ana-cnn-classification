# ANA Classification using CNNs (Transfer Learning)

This repository contains the code used for developing convolutional neural network (CNN) models for automated classification of antinuclear antibody (ANA) indirect immunofluorescence (IIF) images.

Two versions of the models are provided:
- Original (used in manuscript, now legacy)
- Updated (TensorFlow-native implementation without TF Hub)

The models were trained using transfer learning on locally generated HEp-2 IIF images as part of routine clinical diagnostics.

---

## Overview

Two classification tasks are implemented:

1. **Binary classification**

   * ANA-positive vs ANA-negative

2. **Multiclass classification**

   * ANA pattern classification (7 classes):

     * centromere (cen)
     * dense fine speckled (dfs)
     * homogeneous (hom)
     * speckled (kor)
     * nuclear membrane (mem)
     * nuclear dots (nds)
     * nucleolar (nuc)

Original models use a pre-trained **EfficientNetV2-XL (ImageNet-21k-ft1k)** backbone as a fixed feature extractor.

The updated models use a pre-trained **EfficientNetV2-L (ImageNet)** backbone as a fixed feature extractor.

---

## Repository Structure

```
.
├── binary_model__efficientnetv2l.py        # Binary ANA classification
├── binary_model_original.py                # Binary ANA classification
├── multiclass_model_efficientnetv2l.py     # ANA pattern classification
├── multiclass_model_original.py            # ANA pattern classification
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## Requirements

The code for the original models were developed and tested using:

* Python 3.10+
* TensorFlow 2.15
* TensorFlow Hub 0.16

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Data

Due to the use of clinical diagnostic images, the dataset cannot be publicly shared.

* Images originate from routine ANA HEp-2 IIF testing
* Labels reflect routine clinical interpretation

To run the code, users must provide their own dataset structured as:

```
dataset/
├── class_1/
├── class_2/
└── ...
```

---

## Model Description

The original models follow the same general architecture:

* Pre-trained EfficientNetV2-XL feature extractor (frozen)
* Dropout (0.2)
* Dense classification layer with L2 regularization

### Training setup

* Optimizer: Adam (learning rate = 0.001)
* Loss:

  * Binary cross-entropy (binary model)
  * Categorical cross-entropy (multiclass model)
* Batch size: 32
* Train/validation split: 80/20

Class weights were used for the multiclass model to address class imbalance.

---

## Evaluation

Evaluation is performed on independent datasets not used during training.

The scripts include:

* Accuracy calculation
* Per-sample predictions
* Visualization of misclassified images

Binary classification uses a fixed decision threshold of 0.5.

---

## Reproducibility

To ensure reproducibility:

* Fixed random seeds are used
* Model architecture and training parameters are explicitly defined
* The provided scripts reflect the original training pipelines used in the study

Note that exact reproducibility may still vary depending on hardware and environment.

---

## Data and Code Availability

The code used for model development and evaluation is provided in this repository.

Due to data privacy restrictions:

* Raw image data cannot be shared
* De-identified data may be available upon reasonable request, subject to institutional approval

---

## Citation

If you use this code or build upon this work, please cite:

[pending]

---

## Notes

This repository is intended to support transparency and reproducibility of the methods described in the associated manuscript.
It is not intended as a production-ready clinical tool.
