""" 
XGBoost Time Series Model for Process Forecasting

This script:
- Loads pre-split time series data (`train.csv`, `test_final.csv`)
- Applies feature engineering
- Trains both baseline and actor-enriched XGBoost regressors
- Evaluates them via cross-validation and holdout testing
- Visualizes results and interprets with SHAP

Dependencies:
- pandas, numpy, xgboost, scikit-learn, shap, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel, sem, t, wilcoxon
from sklearn.utils import resample
import shap

from models.feature_engineering import feature_engineering

best_xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'gamma': 0,
    'min_child_weight': 1,
    'random_state': 42
}


# Load data from train/test split
from utils.data_utils import load_train_test_split

df_train_full, df_test_final = load_train_test_split("data/final_multivariatetimeseries.csv")


# === Cross-Validation on training set ===
tscv = TimeSeriesSplit(n_splits=5)
rmse_b, mae_b = [], []
rmse_a, mae_a = [], []
all_true, all_pred_b, all_pred_a = [], [], []
all_err_b, all_err_a = [], []
imp_b, imp_a = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(df_train_full)):
    print(f" Fold {fold + 1}")

    df_train_raw = df_train_full.iloc[:test_idx[0]].copy()
    df_test_raw = df_train_full.iloc[test_idx].copy()

    df_train_fe, baseline_features, actor_features = feature_engineering(df_train_raw, residual_target=True)
    df_context = pd.concat([df_train_raw.tail(40), df_test_raw])
    df_test_fe, _, _ = feature_engineering(df_context, residual_target=True)
    df_test_fe = df_test_fe.loc[df_test_raw.index.intersection(df_test_fe.index)]

    df_train_fe.dropna(inplace=True)
    df_test_fe.dropna(inplace=True)

    X_train_b = df_train_fe[baseline_features]
    X_train_a = df_train_fe[baseline_features + actor_features]
    y_train = df_train_fe["target"]

    X_test_b = df_test_fe[baseline_features]
    X_test_a = df_test_fe[baseline_features + actor_features]
    y_test = df_test_fe["target"]

    base_values = df_test_fe['TT'].shift(1).iloc[1:].reset_index(drop=True)
    y_test = y_test.iloc[1:].reset_index(drop=True)
    X_test_b = X_test_b.iloc[1:].reset_index(drop=True)
    X_test_a = X_test_a.iloc[1:].reset_index(drop=True)

    model_b = XGBRegressor(**best_xgb_params)
    model_b.fit(X_train_b, y_train)
    pred_b = model_b.predict(X_test_b) + base_values

    model_a = XGBRegressor(**best_xgb_params)
    model_a.fit(X_train_a, y_train)
    pred_a = model_a.predict(X_test_a) + base_values

    y_true = base_values + y_test

    rmse_b.append(np.sqrt(mean_squared_error(y_true, pred_b)))
    mae_b.append(mean_absolute_error(y_true, pred_b))
    rmse_a.append(np.sqrt(mean_squared_error(y_true, pred_a)))
    mae_a.append(mean_absolute_error(y_true, pred_a))

    all_true.extend(y_true)
    all_pred_b.extend(pred_b)
    all_pred_a.extend(pred_a)
    all_err_b.extend(y_true - pred_b)
    all_err_a.extend(y_true - pred_a)

    imp_b.append(dict(zip(X_train_b.columns, model_b.feature_importances_)))
    imp_a.append(dict(zip(X_train_a.columns, model_a.feature_importances_)))

    if fold == tscv.get_n_splits() - 1:
        xgb_preds = {
            'y_true': y_true,
            'y_pred_b': pred_b,
            'y_pred_a': pred_a,
            'index': df_test_fe.index[1:]
        }

# === Evaluation and Visuals ===
def cohen_d(x, y):
    diff = np.array(x) - np.array(y)
    return diff.mean() / diff.std(ddof=1)

def confidence_interval(data, confidence=0.95):
    m = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2., len(data)-1)
    return f"{m:.3f} ± {h:.3f}"

def summarize_model_comparison(name, rmse_b, rmse_a, mae_b, mae_a):
    stat, p_rmse = ttest_rel(rmse_b, rmse_a)
    return {
        'Model': name,
        'RMSE Baseline': confidence_interval(rmse_b),
        'RMSE Actor': confidence_interval(rmse_a),
        'RMSE Δ': f"{np.mean(rmse_b) - np.mean(rmse_a):.3f}",
        'MAE Baseline': confidence_interval(mae_b),
        'MAE Actor': confidence_interval(mae_a),
        'MAE Δ': f"{np.mean(mae_b) - np.mean(mae_a):.3f}",
        'p-value (RMSE)': f"{p_rmse:.4f}",
        'Cohen’s d': f"{cohen_d(rmse_b, rmse_a):.3f}"
    }

# Print summary
results = [summarize_model_comparison("XGBoost", rmse_b, rmse_a, mae_b, mae_a)]
results_df = pd.DataFrame(results)
print("\n Cross-Validation Model Comparison:")
print(results_df.to_markdown(index=False))

# === Final Holdout Test ===
print("\n Final Holdout Evaluation...")

df_train_fe, baseline_features, actor_features = feature_engineering(df_train_full, residual_target=True)
df_context = pd.concat([df_train_full.tail(40), df_test_final])
df_test_fe, _, _ = feature_engineering(df_context, residual_target=True)
df_test_fe = df_test_fe.loc[df_test_final.index.intersection(df_test_fe.index)]

df_train_fe.dropna(inplace=True)
df_test_fe.dropna(inplace=True)

X_train_b = df_train_fe[baseline_features]
X_train_a = df_train_fe[baseline_features + actor_features]
y_train = df_train_fe["target"]

X_test_b = df_test_fe[baseline_features]
X_test_a = df_test_fe[baseline_features + actor_features]
y_test = df_test_fe["target"]

base_values = df_test_fe["TT"].shift(1).iloc[1:].reset_index(drop=True)
y_test = y_test.iloc[1:].reset_index(drop=True)
X_test_b = X_test_b.iloc[1:].reset_index(drop=True)
X_test_a = X_test_a.iloc[1:].reset_index(drop=True)

final_model_b = XGBRegressor(**best_xgb_params).fit(X_train_b, y_train)
final_model_a = XGBRegressor(**best_xgb_params).fit(X_train_a, y_train)

pred_b = final_model_b.predict(X_test_b) + base_values
pred_a = final_model_a.predict(X_test_a) + base_values
y_true = base_values + y_test

# === Bootstrapped Performance Summary ===
def bootstrap_summary(y_true, y_pred, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    rmse_vals, mae_vals, r2_vals = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        rmse_vals.append(np.sqrt(mean_squared_error(yt, yp)))
        mae_vals.append(mean_absolute_error(yt, yp))
        r2_vals.append(r2_score(yt, yp))
    return {
        'rmse': (np.mean(rmse_vals), np.std(rmse_vals)),
        'mae': (np.mean(mae_vals), np.std(mae_vals)),
        'r2': (np.mean(r2_vals), np.std(r2_vals))
    }

metrics_b = bootstrap_summary(y_true.values, pred_b)
metrics_a = bootstrap_summary(y_true.values, pred_a)

def fmt(mean, std): return f"{mean:.3f} ± {std:.3f}"

summary_test = pd.DataFrame([{
    'Model': 'XGBoost',
    'RMSE Baseline': fmt(*metrics_b['rmse']),
    'RMSE Actor': fmt(*metrics_a['rmse']),
    'RMSE Δ': f"{metrics_b['rmse'][0] - metrics_a['rmse'][0]:.3f}",
    'MAE Baseline': fmt(*metrics_b['mae']),
    'MAE Actor': fmt(*metrics_a['mae']),
    'MAE Δ': f"{metrics_b['mae'][0] - metrics_a['mae'][0]:.3f}",
    'R² Baseline': fmt(*metrics_b['r2']),
    'R² Actor': fmt(*metrics_a['r2']),
    'R² Δ': f"{metrics_a['r2'][0] - metrics_b['r2'][0]:.3f}"
}])

print("\n Final Holdout Results:")
print(summary_test.to_markdown(index=False))

# === SHAP Plots ===
explainer_b = shap.Explainer(final_model_b, X_test_b)
explainer_a = shap.Explainer(final_model_a, X_test_a)

shap.plots.bar(explainer_b(X_test_b), max_display=10, show=True)
shap.plots.bar(explainer_a(X_test_a), max_display=10, show=True)
