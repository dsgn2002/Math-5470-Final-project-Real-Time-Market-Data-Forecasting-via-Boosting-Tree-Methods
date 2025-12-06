# MATH 5470 Final Project

## Real-Time Market Data Forecasting

### Overview

This project tackles the Jane Street Market Prediction challenge, aiming to forecast financial market returns using machine learning. Given the weak linear correlations with the target variable and heterogeneous feature distributions, we employ tree-based gradient boosting models that effectively capture non-linear feature interactions and combine multiple weakly predictive features into stronger ensemble signals.

### Dataset

The dataset consists of anonymized real-world market data with 79 numerical features, 9 lagged responder variables, and associated metadata (date_id, time_id, symbol_id, weight). The target variable (`responder_6`) represents a financial return metric to be predicted.



### Model Architecture


We experimented with two gradient boosting frameworks: LightGBM and XGBoost. Both models offer:

Efficient handling of large-scale tabular data,
Native support for missing values,
Ability to capture complex non-linear relationships and feature interactions and 
Fast training and inference times suitable for real-time prediction
LightGBM uses leaf-wise tree growth for faster training, while XGBoost employs level-wise growth with additional regularization options.



### Evaluation Metric

Performance is measured using **Weighted R² Score**, which accounts for sample importance through the provided weight column:

$$R^2_w = 1 - \frac{\sum w_i (y_i - \hat{y}_i)^2}{\sum w_i (y_i - \bar{y}_w)^2}$$

### Results

| Metric | Validation Score |
|--------|------------------|
| Weighted R² | **0.008247** |

While this R² value may appear small, it represents meaningful predictive power in financial markets where returns are inherently noisy and difficult to forecast. Even modest improvements in prediction accuracy can translate to significant value in real-world trading applications.

### Repository Structure

```
├── lightgbm.ipynb          # Feature preprocessing, EDA, visualization & model training
├── xgboost.ipynb           # xgboost model training
└── README.md               # Project documentation
```

### Requirements

```
polars
lightgbm
numpy
matplotlib
scikit-learn
```



---

