# 🏠 House Price Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Project-House%20Price%20Prediction-green?style=for-the-badge" alt="Project Badge">
  <img src="https://img.shields.io/badge/Methods-Linear%20Regression%20%7C%20Random%20Forest%20%7C%20XGBoost-blue?style=for-the-badge" alt="Methods Badge">
</p>

<p align="center">
  This project builds a regression model to predict house sale prices
  using <b>Multiple Linear Regression</b>, <b>Random Forest</b>, and <b>XGBoost</b>,
  with hyperparameter tuning to find the best model.
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Background](#-background)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Experiment Results](#-experiment-results)
- [Conclusion](#-conclusion)
- [Future Improvement](#-future-improvement)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Technologies Used](#-technologies-used)

---

## 🎯 Project Overview

This project develops a house price prediction system using three regression
algorithms:

- **Multiple Linear Regression** — a simple, interpretable linear baseline
- **Random Forest Regressor** — an ensemble of decision trees that captures
  non-linear patterns
- **XGBoost Regressor** — a gradient boosting model known for strong tabular
  performance

The main goal is to **predict the final sale price of a house** as accurately
as possible based on its physical features, location, and condition.

---

## 🧩 Background

The property market is complex — from global trends down to local Indonesian
conditions. Prices are influenced by a long list of factors: overall quality,
living area, neighborhood, year built, garage size, basement, and many more.

Traditional valuation methods struggle with this complexity. Manual appraisals
are slow, costly, and prone to human bias, while buyers often have no easy
reference for whether an offered price is fair. A machine learning approach
that can process **hundreds of variables simultaneously** offers a way to
produce **objective, fast, and accurate** price predictions.

---

## ⚠️ Problem Statement

Three real problems motivate this project:

| Problem                       | Indicator                                                                 |
| ----------------------------- | ------------------------------------------------------------------------- |
| **Price uncertainty**         | ~63% of buyers are not confident whether the offered price is fair        |
| **Slow conventional process** | ~70% of valuations are manual; estimation error of 10–20% is common       |
| **Lack of transparency**      | 15–30% gap between listing prices and actual transaction prices           |

A data-driven model that learns from real transaction features can directly
address all three.

---

## 🎯 Project Objectives

1. Build a machine learning model that predicts house prices accurately based
   on property features
2. Compare multiple algorithms to find the model with the best performance
3. Identify the key features that most influence house sale price

---

## 📊 Dataset

The dataset used in this project contains house sale transaction records
with detailed property features.

### Dataset Summary

- File used: **`train.csv`**
- Total rows: **1,460 transactions**
- Total columns: **81** (1 ID, 79 features, 1 target)
- Numeric features: **37**
- Categorical features: **43**
- Columns with missing values: **19**

### Target Variable (`SalePrice`)

| Statistic | Value |
| --------- | ---------- |
| Mean      | $180,921   |
| Median    | $163,000   |
| Std Dev   | $79,443    |
| Min       | $34,900    |
| Max       | $755,000   |
| Skewness  | 1.88 (right-skewed) |

The target is right-skewed, so a **log transformation** is applied during
analysis to make the distribution closer to normal (skewness drops to 0.12).

### Main Feature Groups

| Group              | Example Columns |
| ------------------ | --------------- |
| Quality & condition | `OverallQual`, `OverallCond`, `ExterQual`, `KitchenQual`, `BsmtQual` |
| Size & area        | `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `LotArea`, `GarageArea` |
| Rooms              | `FullBath`, `HalfBath`, `BedroomAbvGr`, `TotRmsAbvGrd`, `Fireplaces` |
| Location           | `Neighborhood`, `MSZoning`, `Condition1`, `Condition2` |
| Year & age         | `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `YrSold` |
| Garage             | `GarageType`, `GarageCars`, `GarageArea`, `GarageFinish` |
| Sale info          | `SaleType`, `SaleCondition`, `MoSold` |

### Top 10 Features Correlated with `SalePrice`

| #  | Feature       | Correlation | Description                          |
| -- | ------------- | ----------- | ------------------------------------ |
| 1  | OverallQual   | +0.7910     | Overall material & finishing quality |
| 2  | GrLivArea     | +0.7086     | Above-ground living area             |
| 3  | GarageCars    | +0.6404     | Garage capacity (cars)               |
| 4  | GarageArea    | +0.6234     | Garage area                          |
| 5  | TotalBsmtSF   | +0.6136     | Total basement area                  |
| 6  | 1stFlrSF      | +0.6059     | First floor area                     |
| 7  | FullBath      | +0.5607     | Full bathrooms                       |
| 8  | TotRmsAbvGrd  | +0.5337     | Total rooms above ground             |
| 9  | YearBuilt     | +0.5229     | Year built                           |
| 10 | YearRemodAdd  | +0.5071     | Last remodel year                    |

### Key EDA Findings

- **Quality > Size** — `OverallQual` (r = 0.79) is more predictive than raw
  building size alone
- **Total area dominates** — `GrLivArea` + `TotalBsmtSF` are very strongly
  correlated with price
- **Newer is pricier** — houses with higher `YearBuilt` consistently sell at
  significantly higher prices
- **Garages matter** — `GarageCars` and `GarageArea` are both in the top 5
  most influential features
- **Location effect** — houses near business/popular areas (Northridge,
  Northridge Heights, Stone Brook) average higher sale prices
- **Garage type effect** — built-in garages (under the house) correlate with
  higher-priced homes
- **Outliers** — about 4.18% of houses (61 records) sit outside the IQR range
  for `SalePrice`

---

## 🔬 Methodology

The project workflow consists of these main stages:

1. **Setup & Import** — Import all required libraries
2. **Load Dataset** — Read `train.csv` and inspect its shape and types
3. **Exploratory Data Analysis (EDA)** — Analyze the target distribution,
   correlations, categorical patterns, and outliers
4. **Handle Missing Values**
   - Columns with >80% missing → filled with `"None"` (means "no facility")
   - Numeric columns → filled with **median**
   - Categorical columns → filled with **mode**
   - Result: 0 missing values
5. **Feature Encoding** — Manual label encoding for all 43 categorical features
6. **Feature Engineering** — Create 10 new features:
   - `TotalSF`, `HouseAge`, `RemodelAge`, `TotalBath`, `TotalPorchSF`
   - `HasPool`, `HasGarage`, `HasBsmt`, `HasFireplace`, `QualityScore`
7. **Train-Test Split & Scaling** — 70:30 split, then `StandardScaler` for
   linear models
8. **Modeling** — Train three baseline models: Linear Regression, Random Forest,
   XGBoost
9. **Hyperparameter Tuning**
   - **Ridge / Lasso** with `GridSearchCV` for the linear model
   - **RandomizedSearchCV** for both Random Forest and XGBoost
10. **Model Comparison** — Compare all six models (3 baseline + 3 tuned) on
    the test set
11. **Error Analysis** — Inspect prediction error distribution and accuracy
    bands for the best model

---

## 📈 Experiment Results

All models are evaluated using: **R² Score**, **RMSE**, **MAE**, **MAPE**,
and the **train-test gap** as an overfitting indicator.

### Model Comparison (Test Set)

| Model                       | Train R² | Test R²    | Test RMSE   | Test MAE    | Test MAPE | Overfit Gap |
| --------------------------- | -------- | ---------- | ----------- | ----------- | --------- | ----------- |
| Linear Regression           | 0.8851   | 0.8189     | $35,552     | $20,991     | 12.59%    | 0.0662      |
| Linear Regression (Tuned)   | 0.8519   | 0.8513     | $32,216     | $20,874     | 12.30%    | 0.0006      |
| Random Forest (Default)     | 0.9780   | 0.9065     | $25,543     | $15,485     | 9.30%     | 0.0715      |
| Random Forest (Tuned)       | 0.9665   | 0.8917     | $27,495     | $16,177     | 9.70%     | 0.0749      |
| **★ XGBoost (Default)**     | **0.9972** | **0.9288** | **$22,294** | **$14,357** | **8.68%** | **0.0684**  |
| XGBoost (Tuned)             | 0.9376   | 0.9053     | $25,712     | $15,558     | 9.43%     | 0.0324      |

### Tuning Details

**Linear Regression — Ridge & Lasso (`GridSearchCV`, 5-fold CV)**

- Best Ridge alpha: **100.0** (CV R² = 0.7394)
- Best Lasso alpha: **1.456** (CV R² = 0.7150)
- Selected: **Ridge**

**Random Forest (`RandomizedSearchCV`, 20 iterations × 5-fold CV)**

- Best parameters: `n_estimators=100`, `max_depth=30`, `min_samples_split=5`,
  `min_samples_leaf=1`, `max_features='sqrt'`
- Best CV R²: **0.8442**

**XGBoost (`RandomizedSearchCV`, 30 iterations × 5-fold CV)**

- Best parameters: `n_estimators=500`, `learning_rate=0.01`, `max_depth=4`,
  `min_child_weight=15`, `subsample=0.8`, `colsample_bytree=0.8`,
  `reg_alpha=0.5`, `reg_lambda=1.0`
- Notable: tuning **reduced overfitting** (gap dropped from 0.068 → 0.032),
  but slightly lowered test R² compared to the default

### Error Analysis (Best Model: XGBoost Default)

| Accuracy Band       | Coverage |
| ------------------- | -------- |
| Within 5% error     | 46.12%   |
| Within 10% error    | 74.43%   |
| Within 15% error    | 87.44%   |
| Within 20% error    | 93.15%   |

- Mean Absolute Error: **$14,357**
- Median Absolute Error: **$8,601**
- Mean Absolute Percentage Error: **8.68%**

### Top 10 Feature Importances (XGBoost Default)

| Rank | Feature       | Importance |
| ---- | ------------- | ---------- |
| 1    | OverallQual   | 34.48%     |
| 2    | TotalSF       | 13.78%     |
| 3    | BsmtQual      | 4.91%      |
| 4    | GarageCars    | 4.85%      |
| 5    | FullBath      | 3.85%      |
| 6    | KitchenAbvGr  | 2.80%      |
| 7    | QualityScore  | 2.16%      |
| 8    | HasFireplace  | 2.02%      |
| 9    | TotalBath     | 1.48%      |
| 10   | KitchenQual   | 1.47%      |

The engineered features **`TotalSF`** (rank 2) and **`QualityScore`** (rank 7)
both make the top 10 — clear evidence that feature engineering meaningfully
helped the model.

---

## ✅ Conclusion

1. **Higher quality and larger living area lead to higher prices.** Houses
   located near business and popular areas (Northridge, Northridge Heights,
   Stone Brook) tend to sell at higher average prices.
2. **The best model is XGBoost (Default)**, with:
   - Test R² = **0.9288** (~92.88%)
   - Test MAPE = **8.68%**
   - Overfit gap = **0.068**
3. **Top 3 most important features**:
   - `OverallQual` — Building Quality (34.48%)
   - `TotalSF` — Total Building Area (13.78%)
   - `BsmtQual` — Basement Quality (4.91%)
4. **Building quality is the dominant factor** in determining sale price —
   even more than the building's total area.

---

## 🚀 Future Improvement

The current model already performs well. The next logical step is to move
into **Deployment & Productionization**, so the model can actually be used
in real property-business workflows (e.g. pricing tools for agents, fair-price
checkers for buyers).

---

## 📁 Project Structure

```bash
<root>/
│
├── dataset/
│   ├── train.csv
│   └── data_description.txt
│
├── documentation/
│   └── Machine_Learning_Project_3_House_Price_Prediction__ML_A.pdf
│
├── notebook/
│   └── House_Price_Prediction.ipynb
│
└── README.md
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10 or a compatible version
- Jupyter Notebook or JupyterLab (or Google Colab)
- `pip`

### Install the main libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Optional virtual environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

---

## 💻 How to Run

### 1. Prepare the dataset

Make sure `train.csv` is available inside the `dataset/` folder. The notebook
currently reads from a Google Drive path by default — update the file path
inside the notebook to point to `dataset/train.csv` if you run locally.

### 2. Run the notebook

Open and run:

```bash
notebook/House_Price_Prediction.ipynb
```

The notebook covers all steps from start to finish:

- Data loading and EDA
- Missing value handling and label encoding
- Feature engineering (10 new features)
- Train-test split and feature scaling
- Training Linear Regression, Random Forest, and XGBoost
- Hyperparameter tuning with GridSearchCV (Ridge/Lasso) and RandomizedSearchCV
- Model evaluation, comparison, and error analysis

---

## 🧰 Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **scikit-learn**
- **XGBoost**
- **Jupyter Notebook**
