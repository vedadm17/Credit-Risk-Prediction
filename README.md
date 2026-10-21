# Credit Risk Prediction using Ensemble Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17HFXz0PINXF5nsvJIzhod_KySDe5hmuf#scrollTo=9LXFZvbY55Wb)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end machine learning project evaluating ensemble learning techniques—Random Forest, AdaBoost, and Gradient Boosting—to accurately identify high-risk loan applicants and mitigate financial default risk.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack & Tools](#-tech-stack--tools)
- [Dataset Overview](#-dataset-overview)
- [Project Architecture & Pipeline](#-project-architecture--pipeline)
- [Results & Performance](#-results--performance)
- [Getting Started & How to Run](#-getting-started--how-to-run)
- [Repository Structure](#-repository-structure)
- [License](#-license)

---

## 🔬 Overview
Evaluate and compare ensemble learning models to automate credit scoring and prevent loan defaults in retail banking.

- **Problem Statement:** Financial institutions face substantial monetary loss when granting loans to high-risk applicants who default. Conversely, overly strict criteria result in rejected applications from creditworthy applicants, causing lost interest revenue.
- **Objective:** Build, evaluate, and compare Random Forest, AdaBoost, and Gradient Boosting classifiers to accurately classify borrowers into High Risk (Default) and Low Risk (Non-Default).
- **Key Takeaway:** Addressing severe class imbalance using SMOTE combined with Gradient Boosting yields top predictive performance (ROC-AUC ~0.941), significantly improving default detection without sacrificing overall precision.

---

## ✨ Key Features
- **Data Imputation & Encoding:** Automated median imputation for missing employment length/interest rate values and categorical dummy transformation.
- **Imbalance Management via SMOTE:** Synthetic Minority Over-sampling Technique applied strictly to the training split to prevent data leakage.
- **Comparative Ensemble Pipeline:** Parallel training and benchmarking across Random Forest (Bagging), AdaBoost, and Gradient Boosting (Boosting).
- **Model Interpretability:** Feature importance analysis identifying key financial drivers of credit risk.
- **Visual Analytics:** Automatic generation and visualization of ROC-AUC curve comparisons.

---

## 🛠 Tech Stack & Tools
- **Language:** Python 3.8+
- **Core Libraries:** `pandas`, `numpy`
- **Machine Learning & Imbalance:** `scikit-learn`, `imbalanced-learn` (`imblearn`)
- **Visualization:** `matplotlib`, `seaborn`
- **Environment:** Google Colab / Jupyter Notebooks

---

## 📊 Dataset Overview
- **Source:** Credit Risk Dataset (e.g., German Credit / Kaggle Credit Risk Dataset)
- **Size:** ~32,000 instances × 12 features
- **Target Variable:** `loan_status` (`0`: Non-Default / Low Risk, `1`: Default / High Risk)
- **Key Features:** `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_amnt`, `loan_int_rate`, `cb_person_default_on_file`
- **Preprocessing Steps:**
  1. Missing value imputation using feature medians (`person_emp_length`, `loan_int_rate`).
  2. One-Hot Encoding for categorical features (`person_home_ownership`, `loan_intent`, `loan_grade`).
  3. Feature standardization via `StandardScaler`.

---

## 🏗 Project Architecture & Pipeline

```text
[ Raw Credit Data ] 
       │
       ▼
[ Preprocessing & Cleaning ] ──► Imputation, One-Hot Encoding & Scaling
       │
       ▼
[ Stratified Train-Test Split ] ──► 80% Train / 20% Test Split
       │
       ▼
[ SMOTE Oversampling ] ──► Balanced Training Class Distribution
       │
       ▼
[ Ensemble Model Training ] ──► Random Forest | AdaBoost | Gradient Boosting
       │
       ▼
[ Evaluation & Comparison ] ──► Accuracy, Precision, Recall, F1-Score, ROC-AUC Curves
