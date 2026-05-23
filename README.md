# STD Risk Prediction & Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ML classification pipeline predicting sexually transmitted disease (STD) risk from behavioural and demographic features, supporting public health screening and targeted intervention programmes.

---

## Problem Statement

STD testing rates remain low in many African countries due to stigma and limited access. A risk stratification model enables healthcare workers to prioritise high-risk individuals for testing and counselling, improving resource allocation in resource-constrained settings.

---

## Features

| Feature | Description |
|---------|-------------|
| Multi-Model Classification | Random Forest, Gradient Boosting, Logistic Regression, Decision Tree |
| AUC-ROC Evaluation | Precision-Recall and ROC curve comparison |
| Feature Importance | Key behavioural and demographic risk drivers |
| SMOTE Balancing | Imbalanced class handling for rare positive cases |
| Model Serialisation | Best model saved for clinical screening deployment |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | scikit-learn, imbalanced-learn |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Serialisation | joblib |

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/std-prediction.git
cd std-prediction
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn joblib numpy
python data_generator.py
python eda.py
python train.py
python predict.py
```

---

## Author

**Momah Moses**, Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
