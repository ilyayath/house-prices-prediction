# House Prices Prediction - Kaggle Regression Project

This project is developed for the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) Kaggle competition. The objective is to predict residential home sale prices in Ames, Iowa, using advanced regression techniques and thorough data preprocessing.

## Objective
Build a high-accuracy regression model to predict house prices through:
- Comprehensive exploratory data analysis (EDA) and feature engineering
- Robust preprocessing, including log-transformations, imputation, and scaling
- Hyperparameter optimization using Optuna
- Ensemble modeling with a StackingRegressor

---

## Project Structure

### Exploratory Data Analysis (EDA)
- The target variable (`SalePrice`) is log-transformed to address its skewed, lognormal distribution.
- Numerical feature distributions are visualized using kernel density plots, histograms, and rug plots.
- Outliers are detected using the Interquartile Range (IQR) method and mitigated with `winsorize`.
- Categorical features are analyzed with boxplots, sorted by median `SalePrice` values to identify trends.

### Data Preprocessing
- Log-transformations applied to skewed numerical features (e.g., `LotArea`, `LotFrontage`, `GrLivArea`, `1stFlrSF`).
- Missing values handled with:
  - `None` for meaningful NaNs (e.g., absence of a garage or basement).
  - `KNNImputer` for numerical features to impute based on nearest neighbors.
- Multicollinearity addressed by removing features with correlations above 0.8 (e.g., `TotRmsAbvGrd`, `GarageYrBlt`).
- Low-informative categorical features dropped based on imbalance, low variance, or excessive unique values.
- Categorical variables encoded using one-hot encoding via `pd.get_dummies`.

### Feature Engineering
- New features created to capture additional patterns:
  - `TotalBath`: Sum of full and half bathrooms.
  - `HouseAge`: Years between construction and sale.
  - `YrsSinceRemod`: Years since last remodel.
  - `HasGarage` and `HasPool`: Binary indicators for garage and pool presence.
  - `WasRemodeled`: Binary flag for remodel status.
  - `OverallQualCond`: Interaction term between overall quality and condition.

### Scaling and Feature Selection
- Numerical features scaled using `RobustScaler` to handle outliers effectively.
- Feature importance evaluated with `RandomForestRegressor`, selecting the top 150 features to reduce dimensionality.

---

## Modeling

### Models Used
- `RandomForestRegressor`
- `Ridge`
- `ElasticNet`
- `XGBoost`
- `CatBoost`
- `LGBMRegressor`

### Hyperparameter Tuning
- All models optimized using [Optuna](https://optuna.org/) with 5-fold cross-validation, minimizing RMSE.

### Final Model
- A **StackingRegressor** combines predictions from base models (`XGBoost`, `CatBoost`, `RandomForest`, `LGBM`, `ElasticNet`) with a `Ridge` model as the final estimator:
  ```text
  Final Model = StackingRegressor(estimators=[XGBoost, CatBoost, RandomForest, LGBM, ElasticNet], final_estimator=Ridge)
  ```
- Final predictions blended with `CatBoost` predictions (50:50 ratio) for improved performance.
