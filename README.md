# 💰 Income Level Prediction
### CareerEx × Access Bank YouThrive — Data Science Capstone Project
#### Cohort 3 · Data Science Learning Track · June 2025

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://adewale-income-level-prediction.streamlit.app/)
[![SMOTE](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-purple)](https://imbalanced-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Portfolio](https://img.shields.io/badge/Portfolio-cssadewale.pages.dev-blue)](https://cssadewale.pages.dev)

---

## 🏛️ Programme Context

This project is the **capstone submission** for the **YouThrive Data Science Programme**, a collaborative tech-skills initiative between **CareerEx** and **Access Bank PLC**, delivered as part of the broader YouThrive digital economy initiative.

| Detail | Information |
|--------|-------------|
| **Programme** | YouThrive Data Science — Digital Economy Track |
| **Organiser** | CareerEx |
| **Sponsor / Funder** | Access Bank PLC |
| **Cohort** | Cohort 3 — Data Science Learning Track |
| **Project Timeline** | 1st June – 20th June, 2025 |
| **Capstone Submitted** | July 2025 |
| **Participant** | Adewale Samson Adeagbo |
| **Portfolio** | [cssadewale.pages.dev](https://cssadewale.pages.dev) |

### About CareerEx × Access Bank YouThrive

The **YouThrive Programme** is a flagship Access Bank initiative designed to empower Nigerian youth and MSMEs through capacity development, financial inclusion, and digital economy skills. Access Bank collaborated with **CareerEx** — a structured tech-training platform — to deliver the technology and data science components, addressing Nigeria's significant youth unemployment challenge (approximately 42.5%).

CareerEx's delivery model is built around:
- **Live weekend classes** with industry professionals (not pre-recorded content)
- **Project-based, cohort learning** — participants build real, portfolio-ready products
- **Mentorship and expert feedback** from practitioners working in the field
- **Capstone projects** that simulate real-world job scenarios and are used directly in participants' professional portfolios

Cohort 3 of the YouThrive × CareerEx Data Science track ran in early 2025, training participants in Python, data analysis, machine learning, and model deployment — from fundamentals through to live Streamlit applications.

---

## 📌 Project Overview

This project develops a **supervised binary classification model** to predict whether a U.S. individual earns **above or below \$50,000 per year**, using demographic and occupational data drawn from the 1994 U.S. Census Bureau.

Income classification at this threshold has direct, actionable applications in:
- **Financial product targeting** — identifying underbanked or credit-worthy segments
- **Credit scoring proxies** — informing lending decisions where formal income records are absent
- **Social programme allocation** — directing subsidies and welfare support accurately
- **Policy research** — understanding the structural drivers of income inequality

The problem is intentionally chosen to mirror a real challenge faced across African financial markets: how do you assess income level when formal income documentation is sparse? The approach used in this project — demographic and occupational inference — is directly applicable to financial inclusion work in Nigeria and across the continent.

> **Key modelling challenge:** The dataset exhibits a **3:1 class imbalance** — 75.3% of individuals earn ≤\$50K while only 24.7% earn >\$50K. This makes raw accuracy a misleading success metric, and requires deliberate strategies (SMOTE, ROC-AUC, F1-Score) for fair and honest evaluation.

---

## 📁 Repository Structure

```
income-level-prediction/
│
├── notebook/
│   └── Income_Level_Prediction_FINAL.ipynb   ← Complete end-to-end analysis notebook
│
├── app/
│   └── app.py                                ← Streamlit deployment application
│
├── models/
│   ├── income_prediction_rf_model.joblib     ← Serialised trained Random Forest model
│   └── income_prediction_scaler.joblib       ← Serialised fitted StandardScaler
│
├── data/
│   └── income_data.csv                       ← Dataset (UCI Adult Census)
│
├── requirements.txt                          ← Python dependencies
├── .gitignore                                ← Files excluded from version control
└── README.md                                 ← This file
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository — Adult Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult) |
| **Origin** | 1994 U.S. Census Bureau extraction by Barry Becker |
| **Records** | 48,842 |
| **Raw Features** | 15 (before cleaning and engineering) |
| **Final Features** | 14 (after dropping redundant columns + 2 engineered flags) |
| **Target Variable** | `income_level`: `<=50K` (Class 0) or `>50K` (Class 1) |
| **Class Imbalance** | 75.3% ≤\$50K (31,838 records) · 24.7% >\$50K (10,400 records) |
| **Imbalance Ratio** | Approximately 3:1 |

### Feature Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numerical | Age of individual (range: 17–90) |
| `workclass` | Categorical | Employment sector (Private, Self-emp, Gov, etc.) |
| `education` | Categorical | Highest educational attainment level |
| `marital_status` | Categorical | Marital status (7 categories) |
| `occupation` | Categorical | Job/role type (14 categories) |
| `relationship` | Categorical | Household relationship role |
| `race` | Categorical | Race (5 categories) |
| `sex` | Categorical | Sex (Male / Female) |
| `capital_gain` | Numerical | Investment/capital income received |
| `capital_loss` | Numerical | Investment/capital income lost |
| `hours_per_week` | Numerical | Average weekly working hours |
| `native_country` | Categorical | Country of birth (41 countries) |
| `has_capital_gain` | **Engineered** | Binary flag: 1 if capital_gain > 0, else 0 |
| `has_capital_loss` | **Engineered** | Binary flag: 1 if capital_loss > 0, else 0 |

> **Dropped columns:** `fnlwgt` (census sampling weight — not a predictive feature) and `education_num` (purely redundant with the categorical `education` column).

---

## 🔬 Methodology — End-to-End Pipeline

The project follows a rigorous, documented six-step pipeline. Every decision is grounded in evidence from the data, not assumption.

---

### STEP 1 — Data Loading & Initial Inspection

The raw dataset (48,842 rows × 15 columns) was loaded and subjected to a systematic first look covering:

- **Shape and data types** — verifying expected structure
- **`.info()` and `.describe()`** — confirming dtypes and summary statistics
- **Unique value counts** — revealing the presence of `'?'` as a disguised missing value placeholder in `workclass`, `occupation`, and `native_country`
- **Automated profiling report** — using `ydata_profiling` to generate a full statistical EDA report, including correlations, distributions, and missing value maps

> This step establishes a documented baseline. No data is changed at this stage — only observed.

---

### STEP 2 — Data Cleaning

A sequential, reproducible cleaning pipeline was applied:

| Sub-step | Action | Outcome |
|----------|--------|---------|
| 2a | Column name standardisation | Snake_case, lowercase, consistent format |
| 2b | Data type verification | All types confirmed correct |
| 2c | Unique value inspection | `'?'` placeholder identified in 3 columns |
| 2d | Missing value handling | `'?'` → NaN → mode imputation (categorical) |
| 2e | Outlier detection | Flagged but retained (domain-realistic) |
| 2f | Duplicate removal | Exact duplicates removed |

**Result:** Clean dataset with **zero missing values**, zero duplicates, and fully standardised column names. All cleaning decisions are documented with rationale.

---

### STEP 3 — Exploratory Data Analysis (EDA)

Six data-grounded insights were extracted across three analysis levels:

| Level | Insight | Finding |
|-------|---------|---------|
| Univariate | Class distribution | 31,838 (75.3%) earn ≤\$50K vs. 10,400 (24.7%) earn >\$50K |
| Univariate | Age distribution | Median age 44 for >\$50K group vs. 34 for ≤\$50K — a 10-year seniority gap |
| Bivariate | Education vs income | Doctorate and Prof-school holders show >70% high-income rate |
| Bivariate | Occupation vs income | Exec-managerial and Prof-specialty roles dominate the >\$50K category |
| Bivariate | Capital gain signal | >90% report zero capital gain; non-zero values are a near-decisive predictor of >\$50K |
| Multivariate | Gender and marital status | Both independently stratify income; their interaction compounds the effect |

All EDA visualisations are produced using `matplotlib` and `seaborn`, with plots directly supporting model feature selection decisions in Step 4.

---

### STEP 4 — Preprocessing & Feature Engineering

#### 4a. Feature Engineering
Two binary indicator features were created to extract signal from skewed capital variables:
- `has_capital_gain` = 1 if `capital_gain` > 0 (else 0)
- `has_capital_loss` = 1 if `capital_loss` > 0 (else 0)

This is motivated by the EDA finding that the *presence* of any capital gain/loss is more informative than the amount (>90% of records have zero).

#### 4b. Encoding
- **One-hot encoding** applied to all categorical features using `pd.get_dummies()` with `drop_first=True` to avoid the dummy variable trap
- Result: a fully numerical feature matrix ready for ML algorithms

#### 4c. Train-Test Split
- **Stratified 80/20 split** — preserves the class ratio in both sets
- Random state fixed for reproducibility

#### 4d. Feature Scaling
- `StandardScaler` fitted **on training data only** and applied to both sets
- This is a critical decision: fitting the scaler on the full dataset before splitting would constitute **data leakage**, leading to overly optimistic evaluation results. This project explicitly avoids this error.

#### 4e. SMOTE — Handling Class Imbalance
- **SMOTE (Synthetic Minority Oversampling Technique)** applied to the **training set only** after the split
- Training set grows from ~39,000 to ~47,700 balanced samples
- The test set remains untouched — it reflects real-world imbalance, ensuring honest evaluation

---

### STEP 5 — Model Development & Evaluation

#### 5a–5b. Baseline Model Training & Comparison

Five classifiers were trained and compared across five evaluation metrics:

| Model | ROC-AUC | F1-Score (Weighted) |
|-------|---------|---------------------|
| Logistic Regression | ~0.818 | ~0.823 |
| Decision Tree | ~0.818 | ~0.816 |
| Random Forest | ~0.860 | ~0.861 |
| Gradient Boosting | ~0.850 | ~0.856 |
| Support Vector Machine | ~0.842 | ~0.850 |
| **Tuned Random Forest** | **Best** | **Best** |

**Why Random Forest won:** It achieved the highest ROC-AUC and F1-Score at baseline, indicating strong generalisation without yet being tuned. It is also naturally robust to feature scale differences and handles mixed feature types well.

#### 5c. Hyperparameter Tuning — GridSearchCV

Grid search with **5-fold cross-validation** was run over the following Random Forest hyperparameter space:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

The best parameters were selected based on ROC-AUC score. The tuned model is the project's final recommended model.

#### 5d. Final Model Evaluation

The tuned Random Forest was evaluated on the held-out test set. Exact metric values are printed in `Step 5d` of the notebook. The model outperformed all five baselines across every metric.

#### 5e. Feature Importance Analysis

Top predictive features (by Gini importance):

| Rank | Feature | Interpretation |
|------|---------|---------------|
| 1 | `age` | Career stage and accumulated seniority |
| 2 | `capital_gain` | Raw investment income presence |
| 3 | `hours_per_week` | Proxy for professional role intensity |
| 4 | `marital_status_Married-civ-spouse` | Strongest single categorical predictor |
| 5 | Encoded education & occupation features | Role and qualification level |

#### 5f. Model Serialisation

Both the trained model and fitted scaler are saved using `joblib` for deployment:
- `models/income_prediction_rf_model.joblib`
- `models/income_prediction_scaler.joblib`

The Streamlit app loads these directly — no retraining required at serving time.

---

### STEP 6 — Business Insights & Recommendations

Five actionable recommendations were generated from the model findings:

1. **Financial product targeting** — use the model to identify >50K income segments for premium banking and investment products
2. **Streamlit deployment** — serve predictions via a lightweight web interface accessible to non-technical stakeholders
3. **Fairness auditing** — before commercial use, apply `fairlearn` to audit predictions across sex and race dimensions; the model inherits historical disparities from 1994 data
4. **Data enrichment** — combining Census demographic features with transactional banking data would substantially improve predictive accuracy for a Nigerian or African financial inclusion context
5. **Regression extension** — a follow-on project predicting actual income amount (not just bracket) using Gradient Boosting Regression would complement this classifier

---

## 🏆 Model Performance Summary

| Metric | Performance |
|--------|-------------|
| Algorithm | Random Forest Classifier (Tuned) |
| Tuning Method | GridSearchCV — 5-fold cross-validation |
| Primary Metric | ROC-AUC |
| Class Imbalance Strategy | SMOTE (training set only) |
| Accuracy | Best across all 5 models |
| Precision | Best across all 5 models |
| Recall | Best across all 5 models |
| F1-Score | Best across all 5 models |
| ROC-AUC | Best across all 5 models |

> Exact values are printed in `Step 5d` of the notebook.

---

## 🌐 Live Deployment

The model is deployed as a live Streamlit web application:

**🔗 [adewale-income-level-prediction.streamlit.app](https://adewale-income-level-prediction.streamlit.app/)**

The app allows any user to:
- Input demographic and occupational details via a clean UI
- Receive an instant income level prediction (≤\$50K or >\$50K)
- See the model's confidence score for the prediction

---

## 🚀 Running the Project Locally

### Option 1 — View the Notebook

Open `notebook/Income_Level_Prediction_FINAL.ipynb` in:
- **Jupyter Notebook / JupyterLab** (local install)
- **Google Colab** — upload directly and run cell-by-cell

### Option 2 — Run the Streamlit App Locally

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

**4. Launch the app:**
```bash
streamlit run app/app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Tool / Library | Version | Purpose |
|----------------|---------|---------|
| Python | 3.10+ | Core language |
| pandas | Latest stable | Data manipulation and analysis |
| numpy | Latest stable | Numerical operations |
| matplotlib | Latest stable | Base data visualisation |
| seaborn | Latest stable | Statistical visualisation |
| scikit-learn | 1.3+ | Preprocessing, modelling, evaluation |
| imbalanced-learn | Latest stable | SMOTE oversampling |
| ydata-profiling | Latest stable | Automated EDA profiling report |
| joblib | Latest stable | Model and scaler serialisation |
| Streamlit | Latest stable | Web application deployment |
| Google Colab | — | Notebook development environment |

---

## ⚠️ Limitations & Ethical Notes

**Data vintage:** The UCI Adult Census dataset is drawn from **1994 U.S. Census** data. Labour market dynamics, wage levels, educational attainment patterns, and gender participation rates have all shifted substantially in the intervening 30 years. The model should not be used as a proxy for current income prediction without retraining on contemporary data.

**Inherited bias:** The model learns income stratification patterns that reflect **historical disparities by sex, race, and native country** present in the 1994 data. These patterns are real societal inequalities of that era — the model will reproduce them. A formal fairness audit using `fairlearn` or `aif360` is strongly recommended before any commercial or policy deployment.

**Binary target limitation:** Predicting a ≤/>\$50K threshold captures income bracket — not the actual income amount, income trajectory, or wealth. Two individuals classified identically may have very different financial profiles.

**Missing value imputation:** Mode imputation was applied to `workclass`, `occupation`, and `native_country`. This is a pragmatic choice — the mode values may not represent the true distribution of missing records, which are likely not missing at random.

**Generalisation scope:** The model was trained and evaluated on U.S. Census data. Direct application to Nigerian or African income contexts requires retraining on locally relevant datasets. However, the methodology and pipeline are fully transferable.

---

## 🔗 Related Projects in This Portfolio

This project is part of a broader portfolio of deployed ML and EdTech projects:

| Project | Domain | Algorithm | Live App |
|---------|--------|-----------|----------|
| **Income Level Prediction** *(this project)* | Financial Inclusion | Random Forest + SMOTE | [Live](https://adewale-income-level-prediction.streamlit.app/) |
| TruthLens — Fake News Detector | NLP / Media Literacy | XGBoost + TF-IDF | [Live](https://adewale-fake-news-detector.streamlit.app/) |
| Bank Customer Churn Prediction | Banking / Retention | Gradient Boosting | [Live](https://adewale-bank-customer-churn-prediction.streamlit.app/) |
| Insurance Claim Prediction | Insurance / Risk | Random Forest + SHAP | [Live](https://adewale-insurance-claim-prediction.streamlit.app/) |
| Student At-Risk Predictor | EdTech | Random Forest + SHAP | [Live](https://student-at-risk-predictor.streamlit.app/) |
| NeuroWell — Burnout Rate Predictor | HR / Wellness | Gradient Boosting | [Live](https://adewale-burnout-prediction.streamlit.app/) |
| Yakub Staff Promotion Prediction | HR Analytics | Random Forest | [Live](https://yakub-promotion-prediction.streamlit.app/) |

---

## 👤 Author

**Adewale Samson Adeagbo**

Nigerian Data Scientist, STEM Educator (15+ years, all levels), and AI-Augmented Solutions Developer. Founder of HMG Concepts. Currently enrolled in three simultaneous data science and ML programmes while continuing to teach in a Lagos secondary school.

| Channel | Link |
|---------|------|
| 🌐 Portfolio | [cssadewale.pages.dev](https://cssadewale.pages.dev) |
| 💼 LinkedIn | [linkedin.com/in/adewalesamsonadeagbo](https://linkedin.com/in/adewalesamsonadeagbo) |
| 🐙 GitHub | [github.com/cssadewale](https://github.com/cssadewale) |
| 📺 YouTube | [youtube.com/@hmgconcepts](https://youtube.com/@hmgconcepts) |
| 📧 Email | [hmgconcepts@gmail.com](mailto:hmgconcepts@gmail.com) |
| 💬 WhatsApp | [+234 810 086 6322](https://wa.me/2348100866322) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

---

## 🙏 Acknowledgements

- **CareerEx** — for building a structured, project-first learning environment that produces real portfolio outcomes, not just certificates
- **Access Bank PLC** — for funding and sponsoring the YouThrive programme and making this training accessible to Nigerian youth
- **Mr. Emmanuel Ani** — cohort instructor, whose practical, student-centred teaching approach carried the cohort through to deployment
- **UCI Machine Learning Repository** — for maintaining the Adult Census dataset as a public ML benchmark
- **The Cohort 3 community** — for the shared accountability, peer feedback, and collaborative learning that made this submission stronger

---

*This capstone project represents the completion of the CareerEx × Access Bank YouThrive Data Science Learning Track, Cohort 3. It is one of 12 deployed ML and EdTech projects in a growing portfolio built on the conviction that clear thinking, deliberate learning, and real-world application are the only paths to meaningful capability.*

*"I did not wait to become a data scientist. I built the tools, shipped the models, and let the work speak."*
— Adewale Samson Adeagbo
