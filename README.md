# FraudShield AI — Credit Card Fraud Detection System

A professional, full-stack credit card fraud detection web application built with **Flask**, **Scikit-learn**, **LightGBM**, **XGBoost**, and **Plotly**.

---

## 0. ABSTRACT OF THE PROJECT (10 Points)

1. **Objective:** Design and implement an end-to-end web application (**FraudShield AI**) for real-time and batch credit card transaction fraud detection.
2. **Problem Statement:** Financial fraud datasets exhibit extreme class imbalance (<0.5% fraud cases), rendering naive standard classification algorithms prone to high false-negative rates.
3. **Data Preprocessing & Cleaning:** Vectorized pipelines inspect missing values, impute numeric columns using feature medians, encode categorical attributes, and clean index/PII columns.
4. **Feature Engineering:** Advanced spatio-temporal features are synthesized, including **Haversine geographic distance** between cardholder and merchant, time-of-day/day-of-week indices, log-transformed transaction amounts, and cardholder age.
5. **Exploratory Data Analysis (EDA):** A 9-step interactive Plotly visual analysis module analyzes class distributions, spending behaviors, regional hotspots, customer demographics, and feature correlation matrices.
6. **PCA Feature Identification:** Employs **Principal Component Analysis (PCA)** for dimensionality reduction and identifying top feature load contributions across variance-retained principal components.
7. **Supervised Learning Models:** Trains multiple classification algorithms including **Logistic Regression**, **Linear Model (SGD)**, **Decision Trees**, **Random Forests**, and **Gradient Boosting (LightGBM/XGBoost)** with class imbalance handling.
8. **Unsupervised Learning & Anomaly Detection:** Applies **K-Means Clustering**, **DBSCAN**, and **Isolation Forest Anomaly Detection** to discover non-labelled spatial outliers and anomalous transaction patterns.
9. **Model Accuracy & Evaluation:** Evaluates model performance using **Confusion Matrices**, **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **5-Fold Stratified Cross-Validation**.
10. **Deployment:** Serves model predictions through a responsive **Flask** REST framework, providing Plotly dashboards, batch CSV predictions, result downloads, and single-transaction risk scoring.

---

## 1. DATASET

This app is designed for the **Kaggle Credit Card Fraud Detection** dataset (`fraudTrain.csv` / `fraudTest.csv` with ~1.8 million records).

Expected schema columns include:
```
trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender,
street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time,
merch_lat, merch_long, is_fraud
```

---

## 2. EXPLORATORY DATA ANALYSIS (EDA)

The system includes a 9-step interactive Plotly EDA engine:
1. **Step 1: Dataset Overview** (Missing values, dtypes, shape, duplicates, memory usage)
2. **Step 2: Target Distribution** (Class imbalance pie chart & bar counts)
3. **Step 3: Transaction Amount Analysis** (Log distribution, class boxplots, statistics)
4. **Step 4: Temporal Analysis** (Transactions & fraud rate by hour, day of week, month)
5. **Step 5: Merchant Category Analysis** (Category transaction volume vs fraud rate)
6. **Step 6: Geographic Analysis** (State-wise activity and fraud distribution)
7. **Step 7: Customer Demographics** (Age distribution, gender split, job categories)
8. **Step 8: Correlation Analysis** (Feature correlation heatmap with `is_fraud`)
9. **Step 9: Fraud Patterns & Indicators** (Distance vs amount scatter, hourly fraud heatmaps)

---

## 3. MISSING VALUES IDENTIFICATION & REPLACEMENT

- **Identification:** `step1_overview()` calculates exact null counts and missing percentages across all columns.
- **Replacement:** `clean_data()` applies vectorized median imputation for numerical features and mode imputation for categorical features.

---

## 4. PCA - IDENTIFICATION OF FEATURES FOR MODELING

- **Principal Component Analysis (PCA):** Performs variance-explained dimensionality reduction (retaining 95% variance).
- **Feature Component Loadings:** Extracts the top contributing raw features per Principal Component (PC1, PC2, PC3...).
- **PCA Visual Dashboard:** Displays variance explained ratio per component and cumulative variance percentage.

---

## 5. MACHINE LEARNING MODELS

### Supervised Models
1. **Logistic Regression** (Linear Supervised Classification with L2 regularization)
2. **Linear Model (SGD)** (Stochastic Gradient Descent Linear Model with log loss)
3. **Decision Tree** (Non-linear decision tree classifier with depth control)
4. **Random Forest** (Ensemble of decision trees with balanced class weighting)
5. **LightGBM / XGBoost** (Gradient Boosted Trees with scale_pos_weight)

### Unsupervised Models
1. **K-Means Clustering** (Partitioning transactions into 2 distance clusters; computes silhouette score & fraud alignment)
2. **DBSCAN** (Density-based spatial clustering for discovering noise/outlier transactions)
3. **Isolation Forest** (Tree-partitioning anomaly detection for identifying rare transaction vectors)

---

## 6. MODEL ACCURACY & EVALUATION METRICS

Every supervised model is evaluated on test datasets using:
- **Accuracy Score**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**
- **Confusion Matrix Heatmap**
- **Classification Report**

---

## 7. 5-FOLD CROSS VALIDATION

- **Stratified K-Fold (5 Folds):** Evaluates supervised models across 5 stratified cross-validation folds.
- **Metrics Tracked:** Calculates Mean ± Standard Deviation across all 5 folds for Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## Setup & Installation

### 1. Environment Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Application
```bash
python app.py
```
Open **http://localhost:5000** in your browser.
