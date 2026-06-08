# 🔍 Credit Card Fraud Detection
### End-to-End Machine Learning Pipeline — From Raw Data to Production-Ready Model

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red?style=flat-square)

---

## 🎯 Business Problem

Credit card fraud costs the global financial industry billions of dollars annually.
The core challenge is not just building an accurate model — it is building one that:

- **Catches as many frauds as possible** (high recall), because a missed fraud directly harms a customer
- **Limits false alarms** (reasonable precision), because blocking legitimate transactions erodes user trust
- **Handles extreme class imbalance**, since fraud represents less than 0.2% of all transactions

This project addresses all three challenges through a rigorous, production-oriented ML pipeline.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — whenamancodes/fraud-detection |
| Transactions | ~284,807 |
| Fraud rate | ~0.17% |
| Features | V1–V28 (PCA-anonymized) + Amount + Time |
| Target | Class (0 = Normal, 1 = Fraud) |

> A naive model predicting "no fraud" achieves 99.8% accuracy while being completely useless.
> This is why **Average Precision (AUC-PR)** is the primary metric throughout this project.

---

## ⚙️ Pipeline

### Feature Engineering

| Raw Feature | Transformation | Rationale |
|---|---|---|
| Amount | log1p → Amount_log | Reduces right-skew |
| Time | Extract Hour = (Time/3600) % 24 | Captures intraday patterns |
| Hour | Binary Is_night flag (10 PM – 6 AM) | Fraud risk is higher at night |
| Amount_log, Hour | StandardScaler | Required for linear models |

### Handling Class Imbalance

SMOTE (Synthetic Minority Over-sampling) was selected over random undersampling
because it preserves all original data and synthesizes new fraud examples by
interpolating in feature space.

**Critical implementation detail:** SMOTE is applied exclusively on the training set,
after the train/test split — applying it before would cause data leakage and
artificially inflate evaluation metrics.

### Models Compared

| Model | Strategy |
|---|---|
| Logistic Regression | SMOTE |
| Decision Tree | SMOTE |
| Random Forest | SMOTE |
| Gradient Boosting | SMOTE |
| XGBoost | scale_pos_weight on original data |

### Decision Threshold Optimization

The default 0.5 threshold is rarely optimal for imbalanced problems.
The threshold is swept from 0.1 to 0.9 and the value maximizing F1 is selected,
giving explicit control over the precision/recall trade-off.

---

## 📈 Results

| Model | ROC-AUC | Avg Precision | F1 | Recall |
|---|---|---|---|---|
| XGBoost | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Random Forest | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Gradient Boosting | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Logistic Regression | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Decision Tree | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

> False Negatives are the most costly outcome — a fraud that goes undetected
> directly harms the cardholder. The pipeline minimizes FN through SMOTE,
> model selection, and threshold tuning.

---



## 💡 Key Decisions

**Why Average Precision over ROC-AUC?**
ROC-AUC is optimistic under class imbalance because it accounts for True Negatives,
which are trivially easy when 99.8% of samples are negative. AUC-PR focuses on the
minority class and gives a far more honest picture of real-world performance.

**Why SMOTE over undersampling?**
Undersampling discards legitimate data. SMOTE synthesizes new examples by interpolating
in feature space, allowing the model to learn a richer decision boundary.

**Why tune the decision threshold?**
A missed fraud is far more serious than a false alert. Threshold tuning makes
this trade-off explicit and controllable, rather than hiding it behind a default value.

