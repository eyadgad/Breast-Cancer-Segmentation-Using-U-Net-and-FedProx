# [A Novel Approach to Breast Cancer Segmentation Using U-Net Model with Attention Mechanisms and FedProx](https://doi.org/10.1007/978-3-031-48593-0_23)

---

## Overview
This repository implements a **novel method for breast cancer segmentation** leveraging **Federated Learning** with the **FedProx algorithm** and a **U-Net model enhanced with attention mechanisms**. This study addresses the critical challenge of training on **non-IID distributed medical datasets** while preserving patient privacy. The proposed framework demonstrates high segmentation accuracy on **Ultrasound Breast Cancer Imaging** datasets.

---

## Repository Structure

```
.
├── utils.py                # Utility functions for data handling and augmentation
├── preprocess.py           # Scripts for data preprocessing (normalization, resizing, etc.)
├── models.py               # U-Net implementation with attention mechanisms
├── federated_methods.py    # Implementation of the FedProx algorithm
├── train.py                # Model training in a federated setup
├── evaluate.py             # Model evaluation metrics (Dice Score, IoU, etc.)
├── requirements.txt        # List of required dependencies
```

---

## Results
The proposed framework achieves outstanding performance on the **Ultrasound Breast Cancer Imaging Dataset**:

| Metric        | Value  |
|---------------|--------|
| **Accuracy**  | 96%    |
| **Dice Score**| 0.93   |
| **IoU**       | 0.89   |

These results demonstrate the model's ability to accurately segment tumor regions while preserving privacy.

---

## Prerequisites

### Requirements
- **Python 3.8 or higher**
- **TensorFlow/Keras (2.x or later)**

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

---

## Usage

### Model Training and Data Preprocessing
Train the model in a federated setup using the `FedProx` method:
```bash
python train.py
```

### Evaluation
Evaluate the trained global model using `evaluate.py` to compute metrics such as **Dice Score**, **IoU**, and **Accuracy**:
```bash
python evaluate.py
```

---

## Dataset
- The dataset used in this study consists of **Ultrasound Breast Cancer Imaging Data**.
- For access, please contact [egad@uwo.ca](mailto:egad@uwo.ca).

---

## Key Features

### Federated Learning with FedProx
- The **FedProx algorithm** effectively trains models on **non-IID datasets**, which is common in medical imaging, while addressing issues like data heterogeneity.
- This method enables distributed learning across multiple institutions while preserving privacy.

### Attention-Based U-Net
- **Attention mechanisms** enhance the U-Net model by prioritizing informative regions, ensuring better segmentation, especially for irregular tumor shapes.

### Privacy Preservation
- Federated Learning ensures that no raw medical data is shared between institutions, meeting stringent privacy requirements.

---

## References
If you use this work in your research, please cite:
```plaintext
@InProceedings{10.1007/978-3-031-48593-0_23,
author="Gad, Eyad
and Abou Khatwa, Mustafa
and A. Elattar, Mustafa
and Selim, Sahar",
editor="Waiter, Gordon
and Lambrou, Tryphon
and Leontidis, Georgios
and Oren, Nir
and Morris, Teresa
and Gordon, Sharon",
title="A Novel Approach to Breast Cancer Segmentation Using U-Net Model with Attention Mechanisms and FedProx",
booktitle="Medical Image Understanding and Analysis",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="310--324"
}
```

---

## Contact
For questions, dataset access, or additional details, feel free to contact me, [egad@uwo.ca](mailto:egad@uwo.ca).

---
