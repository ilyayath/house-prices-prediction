# House Prices Prediction - Kaggle Regression Project

This project is based on the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition. The goal is to predict the final sale price of residential homes in Ames, Iowa using advanced regression models and rigorous data processing.

## Goal

To build an accurate regression model for house price prediction using:
- Full EDA and feature engineering
- Robust preprocessing (log-transformations, imputation, scaling)
- Hyperparameter tuning with Optuna
- Ensemble learning via StackingRegressor

---

## Project Structure

### Exploratory Data Analysis (EDA)
- Target variable (`SalePrice`) is log-transformed due to skewed distribution.
- Distributions of numerical features are visualized and summarized.
- Outlier detection via IQR and smoothing with `winsorize`.
- Categorical analysis with boxplots sorted by target medians.

### Data Preprocessing
- Log-transformations applied to key skewed features.
- Missing values handled using:
  - `None` for meaningful NaNs (e.g. no garage).
  - `KNNImputer` for numerical features.
- Multicollinearity removed (features with correlation > 0.8).
- Low-informative categorical features removed (imbalanced, low variation, too many unique values).
- One-hot encoding applied to categorical variables.

### Feature Engineering
- Combined and transformed variables:
  - `TotalBath`, `HouseAge`, `YrsSinceRemod`
  - `HasGarage`, `HasPool`, `WasRemodeled`
  - `OverallQualCond` (interaction term)

### Scaling and Feature Selection
- `RobustScaler` used to normalize numeric features (robust to outliers).
- Feature importance extracted using `RandomForestRegressor`, top 150 features selected.

---

## Modeling

### Models Used:
- `RandomForestRegressor`
- `Ridge`
- `ElasticNet`
- `XGBoost`
- `CatBoost`
- `LGBMRegressor`

### Hyperparameter Tuning:
All models are tuned using [Optuna](https://optuna.org/) with cross-validation (5-fold, RMSE).

### Final Model:
A **StackingRegressor** is built with the above models as base learners and a Ridge model as final estimator:
```text
Final Model = Stacking([XGBoost, CatBoost, RandomForest, LGBM, ElasticNet], final_estimator=Ridge)
