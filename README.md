# 💰 Income Level Prediction
### A Machine Learning Classification Project on U.S. Census Data

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

This project builds a **supervised binary classification model** to predict whether a U.S. individual earns **above or below \$50,000 per year**, using demographic and occupational data from the 1994 U.S. Census Bureau.

The complete end-to-end pipeline covers:
- Systematic data cleaning and inspection
- Exploratory Data Analysis (EDA) with 6 labelled insights
- Feature engineering and preprocessing
- Training and evaluating 5 classification algorithms
- Hyperparameter tuning via GridSearchCV
- Feature importance analysis and model interpretability
- Deployment via a Streamlit web application

> **Key challenge addressed:** The dataset has a **3:1 class imbalance** (75.3% earn ≤\$50K, 24.7% earn >\$50K). SMOTE oversampling and ROC-AUC / F1-Score as primary metrics are used throughout to ensure honest, fair evaluation.

---

## 📁 Repository Structure

```
income-level-prediction/
│
├── notebook/
│   └── Income_Level_Prediction_FINAL.ipynb   ← Complete analysis notebook
│
├── app/
│   └── app.py                                ← Streamlit deployment app
│
├── models/
│   ├── income_prediction_rf_model.joblib     ← Saved trained model
│   └── income_prediction_scaler.joblib       ← Saved fitted scaler
│
├── data/
│   └── income_data.csv                       ← Dataset (UCI Adult Census)
│
├── requirements.txt                          ← Python dependencies
├── .gitignore                                ← Files excluded from Git
└── README.md                                 ← This file
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository — Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult) |
| **Records** | 48,842 |
| **Features** | 14 (after cleaning) |
| **Target** | `income_level`: `<=50K` (0) or `>50K` (1) |
| **Class Imbalance** | 75.3% ≤\$50K · 24.7% >\$50K (ratio ≈ 3:1) |

**Feature Summary:**

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numerical | Age of the individual (17–90) |
| `workclass` | Categorical | Employment type (Private, Gov, Self-emp, etc.) |
| `education` | Categorical | Highest level of education attained |
| `marital_status` | Categorical | Marital status (7 categories) |
| `occupation` | Categorical | Job type (14 categories) |
| `relationship` | Categorical | Relationship role in household |
| `race` | Categorical | Race (5 categories) |
| `sex` | Categorical | Sex (Male / Female) |
| `capital_gain` | Numerical | Investment income gained |
| `capital_loss` | Numerical | Investment income lost |
| `hours_per_week` | Numerical | Weekly working hours |
| `native_country` | Categorical | Country of origin (41 countries) |
| `has_capital_gain` | Engineered | Binary flag: 1 if capital_gain > 0 |
| `has_capital_loss` | Engineered | Binary flag: 1 if capital_loss > 0 |

---

## 🔬 Methodology

### Step 1 — Data Loading & Inspection
Loaded 48,842 records across 15 raw columns. Identified `'?'` placeholders as disguised missing values in `workclass`, `occupation`, and `native_country`.

### Step 2 — Data Cleaning
Sequential cleaning pipeline: column standardisation → dropped `fnlwgt` (sampling weight) and `education_num` (redundant with `education`) → replaced `'?'` with NaN → mode imputation → whitespace stripping → duplicate removal. Result: **clean dataset with 0 missing values**.

### Step 3 — Exploratory Data Analysis
6 data-grounded insights across univariate, bivariate, and multivariate analyses:

| Insight | Finding |
|---------|---------|
| 1 | Class imbalance: 31,838 (75.3%) ≤\$50K vs. 10,400 (24.7%) >\$50K |
| 2 | Age: median 44 for >\$50K vs. 34 for ≤\$50K — a 10-year gap |
| 3 | Education: Doctorate/Prof-school have >70% high-income rate |
| 4 | Occupation: Exec-managerial and Prof-specialty dominate high earners |
| 5 | Capital gain: >90% report zero; non-zero values strongly predict >\$50K |
| 6 | Gender and marital status independently stratify income |

### Step 4 — Preprocessing & Feature Engineering
- Engineered `has_capital_gain` and `has_capital_loss` binary flags
- One-hot encoding with `drop_first=True` to avoid dummy variable trap
- Stratified 80/20 train-test split
- `StandardScaler` — fit on training data only (no data leakage)
- **SMOTE** applied to training set only after the split: balanced training set from ~39K to ~47.7K samples

### Step 5 — Model Development & Evaluation
Trained 5 classifiers, compared on 5 metrics, selected best performer, tuned with GridSearchCV:

| Model | ROC-AUC | F1-Score |
|-------|---------|----------|
| Logistic Regression | ~0.818 | ~0.823 |
| Decision Tree | ~0.818 | ~0.816 |
| Random Forest | ~0.860 | ~0.861 |
| Gradient Boosting | ~0.850 | ~0.856 |
| SVM | ~0.842 | ~0.850 |
| **Tuned Random Forest** | **Best** | **Best** |

### Step 6 — Business Insights & Recommendations
5 actionable recommendations covering financial targeting, Streamlit deployment, fairness auditing, data enrichment, and regression extension.

---

## 🏆 Model Performance

The **Tuned Random Forest Classifier** (selected by GridSearchCV with 5-fold cross-validation) is the final recommended model.

**Key results on held-out test set:**

| Metric | Score |
|--------|-------|
| Accuracy | Best across all models |
| Precision | Best across all models |
| Recall | Best across all models |
| F1-Score | Best across all models |
| **ROC-AUC** | **Best across all models** |

> Exact values are printed in `Step 5d` of the notebook.

**Top 5 Feature Importances:**
1. `age` — career stage and seniority
2. `capital_gain` — investment income presence
3. `hours_per_week` — proxy for professional role intensity
4. `marital_status_Married-civ-spouse` — strongest categorical predictor
5. Education and occupation encoded features

---

## 🚀 Running the Project

### Option 1 — View the Notebook

Open `notebook/Income_Level_Prediction_FINAL.ipynb` in:
- **Jupyter Notebook / JupyterLab** (local)
- **Google Colab** — upload and run directly

### Option 2 — Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/cssadewale/income-level-prediction.git
cd income-level-prediction
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Launch the Streamlit app:**
```bash
streamlit run app/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🌐 Live Demo

> Streamlit deployment link will be added here after deployment.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib / seaborn | Data visualisation |
| scikit-learn | Preprocessing, modelling, evaluation |
| imbalanced-learn | SMOTE oversampling |
| ydata-profiling | Automated EDA report |
| joblib | Model serialisation |
| Streamlit | Web application deployment |

---

## ⚠️ Limitations & Ethical Notes

- Dataset is from **1994 U.S. Census** — labour market patterns have changed substantially
- The model learns income disparities by **sex and race** present in historical data. A formal fairness audit using `fairlearn` is recommended before any commercial deployment
- Binary target (above/below \$50K) does not capture the actual income amount
- Missing values in `workclass`, `occupation`, `native_country` were imputed with mode — not necessarily representative of true values

---

## 👤 Author

**Adewale Adeagbo**
- 📧 buildingmyictcareer@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo)
- 🐙 [GitHub](https://github.com/cssadewale)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*This project is part of a growing data science portfolio demonstrating end-to-end machine learning skills — from raw data through deployed model.*
