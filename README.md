# 🤖 ML-SKILLS

> A hands-on collection of Machine Learning scripts covering the full data science lifecycle — from raw data exploration to model training, evaluation, and interpretability.

---

## 📌 Overview

This repository is a curated set of Python scripts that demonstrate practical Machine Learning skills across multiple domains. Each script is self-contained, well-documented, and produces reproducible outputs (plots, metrics, saved models).

| Topic | Dataset | Script |
|-------|---------|--------|
| EDA & Lifecycle Mapping | Titanic | `code/eda_lifecycle.py` |
| Data Preprocessing Pipeline | Titanic | `code/preprocess.py` |
| Feature Engineering & EDA | Adult Income | `code/fe_eda.py` |
| Logistic Regression | Titanic | `code/logistic_regression.py` |
| Multinomial Logistic Regression | - | `code/multinomial_lr.py` |
| Linear Regression + Scaling | Auto MPG | `code/mpg_regression.py` |
| Regularization (L1/L2/ElasticNet) | - | `code/regularisation.py` |
| Decision Tree | Heart Disease | `code/decision_tree.py` |
| Random Forest | Heart Disease | `code/random_forest.py` |
| Boosting + SHAP / Permutation Importance | Heart Disease | `code/boosting_shap.py` |
| Wine Quality Trees | Wine Quality | `code/wine_trees.py` |

---

## 📁 Repository Structure

```
ML-SKILLS/
├── code/                         # All Python ML scripts
│   ├── preprocess.py             # Titanic preprocessing pipeline
│   ├── eda_lifecycle.py          # EDA & ML lifecycle mapping
│   ├── fe_eda.py                 # Feature engineering + EDA (Adult dataset)
│   ├── logistic_regression.py    # End-to-end logistic regression pipeline
│   ├── multinomial_lr.py         # Multi-class logistic regression
│   ├── mpg_regression.py         # Linear regression with/without scaling
│   ├── regularisation.py         # L1, L2, ElasticNet comparison
│   ├── decision_tree.py          # Decision Tree classifier
│   ├── random_forest.py          # Random Forest ensemble
│   ├── boosting_shap.py          # Gradient Boosting + permutation importance
│   └── wine_trees.py             # Decision Tree on wine quality
│
├── csv/                          # Raw datasets
│   ├── titanic.csv
│   ├── heart.csv
│   ├── diabetes.csv
│   ├── housing.csv
│   ├── adult.csv / adult.data
│   ├── mpg.csv
│   └── winequality.csv
│
├── output/                       # Generated plots, metrics & model artifacts
│   ├── *.png                     # Visualizations
│   ├── *.csv                     # Processed datasets
│   ├── *.txt                     # Metrics & reports
│   └── *.joblib                  # Serialized models/pipelines
│
├── .gitignore
└── README.md
```

---

## 🧠 Scripts Deep-Dive

### 1. 🔍 EDA & Lifecycle Mapping — `eda_lifecycle.py`
Performs end-to-end exploratory data analysis on the **Titanic** dataset.
- Missing value analysis with bar chart
- Survival distribution and rate breakdown
- Survival rate by **Sex** and **Passenger Class**
- Age & Fare distributions (histograms)
- Fare by class (box plot)
- Correlation heatmap for numeric features
- Outputs: `missing_values.png`, `survival_dist.png`, `survival_by_sex_class.png`, `age_fare_dist.png`, `fare_by_class.png`, `correlation_heatmap.png`

---

### 2. ⚙️ Preprocessing Pipeline — `preprocess.py`
Builds a **scikit-learn Pipeline** for the Titanic dataset.
- Feature engineering: `FamilySize`, `IsAlone`
- Median imputation for numeric, mode for categorical
- `StandardScaler` for numerics, `OneHotEncoder` for categoricals
- Saves: `preprocess_pipeline.joblib`, `splits.joblib`, `titanic_cleaned.csv`

---

### 3. 📊 Feature Engineering & EDA — `fe_eda.py`
Deep feature engineering on the **Adult Income** dataset.
- Income distribution, income by education, sex, and hours worked
- Correlation heatmap, scaling comparison
- Outputs engineered dataset as `adult_engineered.csv`

---

### 4. 📈 Logistic Regression — `logistic_regression.py`
Complete end-to-end classification pipeline on **Titanic**.
- `Pipeline` with preprocessing + `LogisticRegression`
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Outputs: `confusion_matrix.png`, `roc_curve.png`, `logistic_model.joblib`, `metrics.txt`

---

### 5. 📉 Linear Regression + Scaling — `mpg_regression.py`
Demonstrates the effect of feature scaling on **Auto MPG** prediction.
- Compares `LinearRegression` **with** and **without** `StandardScaler`
- Shows predictions are identical (affine transform), but coefficient magnitudes differ
- Outputs: `scaling_comparison.png`, `residual_plot.png`, `metrics.txt`

---

### 6. 🌿 Decision Tree — `decision_tree.py`
Trains a `DecisionTreeClassifier` (max_depth=4) on the **Heart Disease** dataset.
- Classification report + accuracy
- Visualizes the tree structure
- Outputs: `tree.png`, `metrics.txt`

---

### 7. 🌲 Random Forest — `random_forest.py`
Trains a 200-estimator `RandomForestClassifier` with OOB scoring on **Heart Disease**.
- Feature importance bar chart
- Confusion matrix
- Outputs: `feature_importance.png`, `confusion_matrix.png`, `metrics.txt`

---

### 8. 🚀 Gradient Boosting + Interpretability — `boosting_shap.py`
Compares `GradientBoostingClassifier` vs `RandomForestClassifier` on **Heart Disease**.
- Selects the best model by ROC-AUC
- Uses **Permutation Importance** as a model-agnostic interpretability tool
- Outputs: `permutation_importance.png`, `metrics.txt`

---

## 📦 Datasets

| File | Description | Size |
|------|-------------|------|
| `titanic.csv` | Titanic passenger survival | ~60 KB |
| `heart.csv` | Heart disease classification | ~18 KB |
| `diabetes.csv` | Pima Indians diabetes | ~97 KB |
| `housing.csv` | California housing prices | ~1.8 MB |
| `adult.csv` | Adult income census | ~3.1 MB |
| `mpg.csv` | Auto fuel efficiency | ~16 KB |
| `winequality.csv` | Wine quality ratings | ~91 KB |

---

## 🛠️ Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

> **Optional (for enhanced boosting):** `xgboost`, `lightgbm`, `shap`

---

## 🚀 How to Run

All scripts must be run from the **project root** directory:

```bash
# EDA
python code/eda_lifecycle.py

# Preprocessing
python code/preprocess.py

# Logistic Regression
python code/logistic_regression.py

# Linear Regression
python code/mpg_regression.py

# Decision Tree
python code/decision_tree.py

# Random Forest
python code/random_forest.py

# Gradient Boosting + Importance
python code/boosting_shap.py
```

All outputs are saved to the `output/` directory automatically.

---

## 📊 Sample Outputs

| Output | Description |
|--------|-------------|
| `roc_curve.png` | ROC curve with AUC score |
| `confusion_matrix.png` | Predicted vs actual classification |
| `feature_importance.png` | Random Forest feature ranking |
| `permutation_importance.png` | Model-agnostic feature importance |
| `tree.png` | Decision Tree visualization |
| `correlation_heatmap.png` | Feature correlation matrix |
| `scaling_comparison.png` | Coefficient magnitudes with/without scaling |
| `residual_plot.png` | Regression residuals |
| `logistic_model.joblib` | Serialized logistic regression model |
| `preprocess_pipeline.joblib` | Serialized preprocessing pipeline |

---

## 🧩 Key ML Concepts Covered

- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering (derived features, encoding)
- ✅ Data Preprocessing Pipelines (`sklearn.Pipeline`)
- ✅ Classification (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- ✅ Regression (Linear Regression, regularization)
- ✅ Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, RMSE, R²)
- ✅ Feature Importance & Model Interpretability
- ✅ Effect of Feature Scaling on Coefficients
- ✅ Ensemble Methods (Bagging, Boosting)

---

## 👤 Author

**Venkatesh Addanki**  
KLH University — CSE Department  
📧 [venkateshaddanki287-eng](https://github.com/venkateshaddanki287-eng)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
