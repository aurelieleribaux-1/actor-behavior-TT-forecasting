"""
LightGBM Time Series Model for Process Forecasting

This script:
- Loads pre-split time series data (`train.csv`, `test_final.csv`)
- Applies feature engineering using historical TT and actor behavior
- Trains both baseline and actor-enriched LightGBM regressors
- Evaluates models via cross-validation and final holdout testing
- Visualizes errors, predictions, and interprets using SHAP

Dependencies:
- pandas, numpy, lightgbm, scikit-learn, shap, matplotlib, seaborn, scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel, sem, t, wilcoxon
from sklearn.utils import resample
import shap

from models.feature_engineering import feature_engineering

# Load data
df_train_full = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
df_test_final = pd.read_csv("data/test_final.csv", index_col=0, parse_dates=True)

# Best parameters
best_lgbm_params = {
    'n_estimators': 1500,
    'learning_rate': 0.05,
    'max_depth': 5,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.9,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'min_child_samples': 1,
    'random_state': 42
}

# Metrics utils
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

def bootstrap_summary(y_true, y_pred, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    rmse_vals, mae_vals, r2_vals = [], [], []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        rmse_vals.append(np.sqrt(mean_squared_error(yt, yp)))
        mae_vals.append(mean_absolute_error(yt, yp))
        r2_vals.append(r2_score(yt, yp))
    return {
        'rmse': (np.mean(rmse_vals), np.std(rmse_vals)),
        'mae': (np.mean(mae_vals), np.std(mae_vals)),
        'r2': (np.mean(r2_vals), np.std(r2_vals))
    }

def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"

if __name__ == "__main__":
    tscv = TimeSeriesSplit(n_splits=5)
    df = df_train_full.copy()

    rmse_b, mae_b, rmse_a, mae_a = [], [], [], []
    all_true, all_pred_b, all_pred_a = [], [], []
    all_err_b, all_err_a = [], []
    imp_b, imp_a = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        print(f" Fold {fold + 1}")
        df_train_raw = df.iloc[:test_idx[0]].copy()
        df_test_raw = df.iloc[test_idx].copy()

        df_train_fe, baseline_features, actor_features = feature_engineering(df_train_raw)
        df_context = pd.concat([df_train_raw.tail(40), df_test_raw])
        df_test_fe, _, _ = feature_engineering(df_context)
        df_test_fe = df_test_fe.loc[df_test_raw.index.intersection(df_test_fe.index)]

        df_train_fe.dropna(inplace=True)
        df_test_fe.dropna(inplace=True)

        X_train_b = df_train_fe[baseline_features]
        X_train_a = df_train_fe[baseline_features + actor_features]
        y_train = df_train_fe['target']

        X_test_b = df_test_fe[baseline_features]
        X_test_a = df_test_fe[baseline_features + actor_features]
        y_test = df_test_fe['target']
        base_values = df_test_fe['TT'].shift(1).iloc[1:].reset_index(drop=True)
        y_test = y_test.iloc[1:].reset_index(drop=True)
        X_test_b = X_test_b.iloc[1:].reset_index(drop=True)
        X_test_a = X_test_a.iloc[1:].reset_index(drop=True)

        model_b = LGBMRegressor(**best_lgbm_params)
        model_b.fit(X_train_b, y_train)
        pred_b = model_b.predict(X_test_b) + base_values

        model_a = LGBMRegressor(**best_lgbm_params)
        model_a.fit(X_train_a, y_train)
        pred_a = model_a.predict(X_test_a) + base_values

        y_true = base_values + y_test

        rmse_b.append(np.sqrt(mean_squared_error(y_true, pred_b)))
        rmse_a.append(np.sqrt(mean_squared_error(y_true, pred_a)))
        mae_b.append(mean_absolute_error(y_true, pred_b))
        mae_a.append(mean_absolute_error(y_true, pred_a))

        all_true.extend(y_true)
        all_pred_b.extend(pred_b)
        all_pred_a.extend(pred_a)
        all_err_b.extend(y_true - pred_b)
        all_err_a.extend(y_true - pred_a)

        imp_b.append(dict(zip(X_train_b.columns, model_b.feature_importances_)))
        imp_a.append(dict(zip(X_train_a.columns, model_a.feature_importances_)))

        if fold == tscv.get_n_splits() - 1:
            lgb_preds = {
                'y_true': y_true,
                'y_pred_b': pred_b,
                'y_pred_a': pred_a,
                'index': df_test_fe.index[1:]
            }

    print("\n RMSE Baseline:", rmse_b)
    print(" RMSE Actor   :", rmse_a)
    print(" MAE Baseline:", mae_b)
    print(" MAE Actor   :", mae_a)

    results = [summarize_model_comparison("LightGBM", rmse_b, rmse_a, mae_b, mae_a)]
    print("\n Summary Table:")
    print(pd.DataFrame(results).to_markdown(index=False))

    # Final holdout evaluation
    print("\n Final Holdout Test...")

    df_train_fe, baseline_features, actor_features = feature_engineering(df_train_full)
    df_context = pd.concat([df_train_full.tail(40), df_test_final])
    df_test_fe, _, _ = feature_engineering(df_context)
    df_test_fe = df_test_fe.loc[df_test_final.index.intersection(df_test_fe.index)]

    df_train_fe.dropna(inplace=True)
    df_test_fe.dropna(inplace=True)

    X_train_b = df_train_fe[baseline_features]
    X_train_a = df_train_fe[baseline_features + actor_features]
    y_train = df_train_fe['target']

    X_test_b = df_test_fe[baseline_features]
    X_test_a = df_test_fe[baseline_features + actor_features]
    y_test = df_test_fe['target']

    base_values = df_test_fe['TT'].shift(1).iloc[1:].reset_index(drop=True)
    y_test = y_test.iloc[1:].reset_index(drop=True)
    X_test_b = X_test_b.iloc[1:].reset_index(drop=True)
    X_test_a = X_test_a.iloc[1:].reset_index(drop=True)

    model_b = LGBMRegressor(**best_lgbm_params).fit(X_train_b, y_train)
    model_a = LGBMRegressor(**best_lgbm_params).fit(X_train_a, y_train)

    pred_b = model_b.predict(X_test_b) + base_values
    pred_a = model_a.predict(X_test_a) + base_values
    y_true = base_values + y_test

    metrics_b = bootstrap_summary(y_true.values, pred_b)
    metrics_a = bootstrap_summary(y_true.values, pred_a)

    summary_test = pd.DataFrame([{
        'Model': 'LightGBM',
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

    print("\n Final Holdout Test Results:")
    print(summary_test.to_markdown(index=False))

    # SHAP interpretability
    explainer_b = shap.Explainer(model_b, X_test_b)
    explainer_a = shap.Explainer(model_a, X_test_a)

    shap.plots.bar(explainer_b(X_test_b), max_display=6, show=True)
    plt.title("SHAP: Top Baseline Features"); plt.show()

    shap.plots.bar(explainer_a(X_test_a), max_display=6, show=True)
    plt.title("SHAP: Top Actor-Enriched Features"); plt.show()
