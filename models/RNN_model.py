"""
RNN with Attention for Time Series Process Forecasting

This script:
- Loads train/test split data (`train.csv`, `test_final.csv`)
- Applies feature engineering
- Trains LSTM and GRU (baseline & actor-enriched) with cross-validation
- Evaluates models on final unseen holdout test set
- Reports metrics, plots predictions, and computes permutation importance

Dependencies:
- pandas, numpy, tensorflow, scikit-learn, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, Bidirectional,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import os
import random

#RNN with attention.py
from feature_engineering import feature_engineering

#load data 
# Load data from train/test split
from utils.data_utils import load_train_test_split

df_train_full, df_test_final = load_train_test_split("data/final_multivariatetimeseries.csv")

# -----------------------------
# Helper Functions
# -----------------------------

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

set_seed(42)

def create_seq_data(df, features, time_steps=15, target_col='target'):
    X, y = [], []
    for i in range(time_steps, len(df) - 1):
        X.append(df[features].iloc[i - time_steps:i].values)
        y.append(df[target_col].iloc[i + 1])
    return np.array(X), np.array(y)

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

    rmse_improvement = np.mean(rmse_b) - np.mean(rmse_a)
    mae_improvement = np.mean(mae_b) - np.mean(mae_a)

    return {
        'Model': name,
        'RMSE Baseline': confidence_interval(rmse_b),
        'RMSE Actor': confidence_interval(rmse_a),
        'RMSE Δ': f"{rmse_improvement:.3f}",  # ← RMSE improvement
        'MAE Baseline': confidence_interval(mae_b),
        'MAE Actor': confidence_interval(mae_a),
        'MAE Δ': f"{mae_improvement:.3f}",    # ← MAE improvement
        'p-value (RMSE)': f"{p_rmse:.4f}",
        'Cohen’s d': f"{cohen_d(rmse_b, rmse_a):.3f}"
    }

def train_attention_rnn(df, features, rnn_type='gru', time_steps=15, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_list, mae_list = [], []
    final_preds = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        print(f"-Fold {fold + 1}")

        df_train_raw = df.iloc[:test_idx[0]].copy()
        df_test_raw = df.iloc[test_idx].copy()

       
        df_train_fe, _, _ = feature_engineering(df_train_raw, residual_target=True)
        df_context = pd.concat([df_train_raw.tail(40), df_test_raw])
        df_test_fe, _, _ = feature_engineering(df_context, residual_target=True)
        df_test_fe = df_test_fe.loc[df_test_raw.index.intersection(df_test_fe.index)]

        df_train_fe.dropna(inplace=True)
        df_test_fe.dropna(inplace=True)

        # Scaling
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        df_train_fe[features] = scaler_x.fit_transform(df_train_fe[features])
        df_test_fe[features] = scaler_x.transform(df_test_fe[features])
        df_train_fe['target_scaled'] = scaler_y.fit_transform(df_train_fe[['target']])
        df_test_fe['target_scaled'] = scaler_y.transform(df_test_fe[['target']])

        # Sequence data
        X_train, y_train = create_seq_data(df_train_fe, features, time_steps, 'target_scaled')
        X_test, y_test = create_seq_data(df_test_fe, features, time_steps, 'target_scaled')

        base_values = df_test_fe['TT'].iloc[time_steps + 1: time_steps + 1 + len(y_test)].reset_index(drop=True)

        # Build model
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = MaxPooling1D(pool_size=2)(x)
        x = Bidirectional(GRU(128, return_sequences=True) if rnn_type == 'gru' else LSTM(128, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = (GRU(64, return_sequences=True) if rnn_type == 'gru' else LSTM(56, return_sequences=True))(x)
    
        x = GlobalAveragePooling1D()(x)
        output = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=output)

        model.compile(optimizer='adam', loss='mse')
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
        ]
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=100, batch_size=32, callbacks=callbacks, verbose=0)

        # Predictions
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        y_pred_final = base_values + y_pred
        y_true_final = base_values + y_test_inv

        rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
        mae = mean_absolute_error(y_true_final, y_pred_final)
        rmse_list.append(rmse)
        mae_list.append(mae)

        if fold == n_splits - 1:
            final_preds = {
                'index': df_test_fe.index[time_steps + 1: time_steps + 1 + len(y_pred)],
                'y_true': y_true_final,
                'y_pred': y_pred_final
            }

    return rmse_list, mae_list, final_preds
# -------------------------
#  Train All RNN Models
# -------------------------
df = df_train_full.copy()
df_model, baseline_features, actor_features = feature_engineering(df, residual_target=True)
set_seed(42)

print("\n LSTM (Baseline)")
rmse_b_lstm, mae_b_lstm, preds_b_lstm = train_attention_rnn(df, baseline_features, rnn_type='lstm')

print(" LSTM (Actor-Enriched)")
rmse_a_lstm, mae_a_lstm, preds_a_lstm = train_attention_rnn(df, baseline_features + actor_features, rnn_type='lstm')

print(" GRU (Baseline)")
rmse_b_gru, mae_b_gru, preds_b_gru = train_attention_rnn(df, baseline_features, rnn_type='gru')

print(" GRU (Actor-Enriched)")
rmse_a_gru, mae_a_gru, preds_a_gru = train_attention_rnn(df, baseline_features + actor_features, rnn_type='gru')


# ===============================
#  Add RNN Results to Global Table
# ===============================
results = [] if 'results' not in globals() else results

results.append(summarize_model_comparison("LSTM", rmse_b_lstm, rmse_a_lstm, mae_b_lstm, mae_a_lstm))
results.append(summarize_model_comparison("GRU", rmse_b_gru, rmse_a_gru, mae_b_gru, mae_a_gru))

#  Show updated comparison
results_df = pd.DataFrame(results)
print("\nFinal Model Comparison Table:")
print(results_df.to_markdown())

# ===============================
#  Final Report
# ===============================
for r in results[-2:]:
    print(f"\n {r['Model']} Summary")
    for k, v in r.items():
        if k != 'Model':
            print(f"  {k}: {v}")


# Optional final full table
if len(results) > 1:
    results_df = pd.DataFrame(results)
    print("\n Final Model Comparison Table:")
    print(results_df.to_markdown())


# ===============================
#  Plot Actual vs Predictions
# ===============================
plt.figure(figsize=(14, 6))
time_steps = 15
plt.plot(df_model.index[time_steps:], df_model['TT'].values[time_steps:], label='Full Actual', color='black')
plt.plot(preds_b_lstm['index'], preds_b_lstm['y_pred'], '--', label='LSTM Baseline', color='blue')
plt.plot(preds_a_lstm['index'], preds_a_lstm['y_pred'], '--', label='LSTM Actor-Enriched', color='green')
plt.plot(preds_b_gru['index'], preds_b_gru['y_pred'], '--', label='GRU Baseline', color='orange')
plt.plot(preds_a_gru['index'], preds_a_gru['y_pred'], '--', label='GRU Actor-Enriched', color='red')
plt.axvspan(preds_b_lstm['index'][0], preds_b_lstm['index'][-1], color='gray', alpha=0.1, label='Test Range')
plt.title(" Train + Test: Actual vs Predictions")
plt.xlabel("Time")
plt.ylabel("Duration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
#  Overlay Plot
# ===============================
plt.figure(figsize=(14, 6))
plt.plot(preds_b_lstm['index'], preds_b_lstm['y_true'], label='Actual', color='black')
plt.plot(preds_b_lstm['index'], preds_b_lstm['y_pred'], '--', label='LSTM Baseline', color='blue')
plt.plot(preds_a_lstm['index'], preds_a_lstm['y_pred'], '--', label='LSTM Actor-Enriched', color='green')
plt.plot(preds_b_gru['index'], preds_b_gru['y_pred'], '--', label='GRU Baseline', color='orange')
plt.plot(preds_a_gru['index'], preds_a_gru['y_pred'], '--', label='GRU Actor-Enriched', color='red')
plt.title(" Comparison: LSTM vs GRU with Attention")
plt.xlabel("Time")
plt.ylabel("Duration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Final Holdout Test ===
print("\n Final Holdout Evaluation...")

# -------------------------------------------
# Seed Setup
# -------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

set_seed(42)

def create_seq_data(df, features, time_steps=15, target_col='target'):
    X, y = [], []
    for i in range(time_steps, len(df) - 1):
        X.append(df[features].iloc[i - time_steps:i].values)
        y.append(df[target_col].iloc[i + 1])
    return np.array(X), np.array(y)


from sklearn.utils import resample

def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, seed=42):
    rmse_vals, mae_vals, r2_vals = [], [], []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    rng = np.random.RandomState(seed)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt_bs = y_true[idx]
        yp_bs = y_pred[idx]
        rmse_vals.append(np.sqrt(mean_squared_error(yt_bs, yp_bs)))
        mae_vals.append(mean_absolute_error(yt_bs, yp_bs))
        r2_vals.append(r2_score(yt_bs, yp_bs))

    def summary(vals):
        return np.mean(vals), np.std(vals)


    return {
        'rmse': summary(rmse_vals),
        'mae': summary(mae_vals),
        'r2': summary(r2_vals)
    }

# -------------------------------------------
# Final Evaluation Function
# -------------------------------------------
def final_evaluate_rnn(df_train, df_test, features, rnn_type='gru', time_steps=15):
    context_window = 50  # Ensure enough rows for lag features
    df_context = pd.concat([df_train.tail(context_window), df_test])

    df_train_fe, _, _ = feature_engineering(df_train, residual_target=True)
    df_test_fe, _, _ = feature_engineering(df_context, residual_target=True)

    #  Keep only rows from actual test period
    df_test_fe = df_test_fe.loc[df_test_fe.index.intersection(df_test.index)]
    df_train_fe.dropna(inplace=True)
    df_test_fe.dropna(inplace=True)

    # Scale features and target
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    df_train_fe[features] = scaler_x.fit_transform(df_train_fe[features])
    df_test_fe[features] = scaler_x.transform(df_test_fe[features])
    df_train_fe['target_scaled'] = scaler_y.fit_transform(df_train_fe[['target']])
    df_test_fe['target_scaled'] = scaler_y.transform(df_test_fe[['target']])

    # Sequence creation
    X_train, y_train = create_seq_data(df_train_fe, features, time_steps, 'target_scaled')
    X_test, y_test = create_seq_data(df_test_fe, features, time_steps, 'target_scaled')

    # Match predicted output strictly to df_test index
    max_len = min(len(y_test), len(df_test))
    index_values = df_test.index[:max_len]
    base_values = df_test['TT'].iloc[:max_len].reset_index(drop=True)

    # Build model
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)

    if rnn_type == 'lstm':
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=True)(x)
    else:
        x = Bidirectional(GRU(128, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = GRU(32, return_sequences=True)(x)


    x = GlobalAveragePooling1D()(x)
    output = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # Predict and inverse-transform
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Trim to match index
    y_pred = y_pred[:max_len]
    y_test_inv = y_test_inv[:max_len]

    y_pred_final = base_values + y_pred
    y_true_final = base_values + y_test_inv

    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    mae = mean_absolute_error(y_true_final, y_pred_final)
    r2 = r2_score(y_true_final, y_pred_final)

    print(f" Final Evaluation on df_test_final ({rnn_type.upper()}):")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")
    print(f"  R²  : {r2:.3f}")


    return {
        'index': index_values,
        'y_true': y_true_final,
        'y_pred': y_pred_final,
        'metrics': bootstrap_metrics(y_true_final, y_pred_final, seed=42),
        'model': model,
        'X_test': X_test,
        'y_test_true': y_test_inv,
        'y_test_scaled': y_test,
        'scaler_y': scaler_y,
        'features': features
    }



def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"

# -------------------------------------------
# Final Test Set Evaluations
# -------------------------------------------
df = df_train_full.copy()
df_model, baseline_features, actor_features = feature_engineering(df, residual_target=True)
set_seed(42)
preds_final_lstm_b = final_evaluate_rnn(df_train_full, df_test_final, baseline_features, rnn_type='lstm')
preds_final_lstm_a = final_evaluate_rnn(df_train_full, df_test_final, baseline_features + actor_features, rnn_type='lstm')
preds_final_gru_b = final_evaluate_rnn(df_train_full, df_test_final, baseline_features, rnn_type='gru')
preds_final_gru_a = final_evaluate_rnn(df_train_full, df_test_final, baseline_features + actor_features, rnn_type='gru')

# -------------------------------------------
# Plot Predictions
# -------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(preds_final_lstm_b['index'], preds_final_lstm_b['y_true'], label='Actual', color='black')
plt.plot(preds_final_lstm_b['index'], preds_final_lstm_b['y_pred'], '--', label='LSTM Baseline', color='blue')
plt.plot(preds_final_lstm_a['index'], preds_final_lstm_a['y_pred'], '--', label='LSTM Actor-Enriched', color='green')
plt.plot(preds_final_gru_b['index'], preds_final_gru_b['y_pred'], '--', label='GRU Baseline', color='orange')
plt.plot(preds_final_gru_a['index'], preds_final_gru_a['y_pred'], '--', label='GRU Actor-Enriched', color='red')
plt.title("🔍 Final Evaluation: Actual vs Predictions")
plt.xlabel("Time")
plt.ylabel("Duration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------
# Final Metric Summary Table
# -------------------------------------------
def extract_metrics(metrics):
    rmse_m, rmse_s = metrics['rmse']
    mae_m, mae_s = metrics['mae']
    r2_m, r2_s = metrics['r2']
    return rmse_m, rmse_s, mae_m, mae_s, r2_m, r2_s

rb_lstm, rs_lstm, mb_lstm, ms_lstm, r2b_lstm, r2s_lstm = extract_metrics(preds_final_lstm_b['metrics'])
ra_lstm, ras_lstm, ma_lstm, mas_lstm, r2a_lstm, r2as_lstm = extract_metrics(preds_final_lstm_a['metrics'])
rb_gru, rs_gru, mb_gru, ms_gru, r2b_gru, r2s_gru = extract_metrics(preds_final_gru_b['metrics'])
ra_gru, ras_gru, ma_gru, mas_gru, r2a_gru, r2as_gru = extract_metrics(preds_final_gru_a['metrics'])

summary_test = pd.DataFrame([
    {
        'Model': 'LSTM',
        'RMSE Baseline': fmt(rb_lstm, rs_lstm),
        'RMSE Actor': fmt(ra_lstm, ras_lstm),
        'RMSE Δ': f"{rb_lstm - ra_lstm:.3f}",
        'MAE Baseline': fmt(mb_lstm, ms_lstm),
        'MAE Actor': fmt(ma_lstm, mas_lstm),
        'MAE Δ': f"{mb_lstm - ma_lstm:.3f}",
        'R² Baseline': fmt(r2b_lstm, r2s_lstm),
        'R² Actor': fmt(r2a_lstm, r2as_lstm),
        'R² Δ': f"{r2a_lstm - r2b_lstm:.3f}",
    },
    {
        'Model': 'GRU',
        'RMSE Baseline': fmt(rb_gru, rs_gru),
        'RMSE Actor': fmt(ra_gru, ras_gru),
        'RMSE Δ': f"{rb_gru - ra_gru:.3f}",
        'MAE Baseline': fmt(mb_gru, ms_gru),
        'MAE Actor': fmt(ma_gru, mas_gru),
        'MAE Δ': f"{mb_gru - ma_gru:.3f}",
        'R² Baseline': fmt(r2b_gru, r2s_gru),
        'R² Actor': fmt(r2a_gru, r2as_gru),
        'R² Δ': f"{r2a_gru - r2b_gru:.3f}",
    }
])


print("\n Final Holdout Test Results Summary:")
print(summary_test.to_markdown(index=False))

def permutation_importance_rnn(model, X_val, y_val_true, y_val_scaled, scaler_y, features, n_repeats=5):
    importances = {}
    baseline_preds = model.predict(X_val).flatten()
    baseline_preds_inv = scaler_y.inverse_transform(baseline_preds.reshape(-1, 1)).flatten()
    baseline_rmse = np.sqrt(mean_squared_error(y_val_true, baseline_preds_inv))

    for i, feature in enumerate(features):
        permuted_rmses = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            np.random.shuffle(X_perm[:, :, i])  # Permute values of feature i across time steps
            perm_preds = model.predict(X_perm).flatten()
            perm_preds_inv = scaler_y.inverse_transform(perm_preds.reshape(-1, 1)).flatten()
            perm_rmse = np.sqrt(mean_squared_error(y_val_true, perm_preds_inv))
            permuted_rmses.append(perm_rmse - baseline_rmse)
        importances[feature] = np.mean(permuted_rmses)

    return pd.Series(importances).sort_values(ascending=False)


# Run permutation importance for actor-enriched LSTM
fi_lstm_a = permutation_importance_rnn(
    preds_final_lstm_a['model'],
    preds_final_lstm_a['X_test'],
    preds_final_lstm_a['y_test_true'],
    preds_final_lstm_a['y_test_scaled'],
    preds_final_lstm_a['scaler_y'],
    preds_final_lstm_a['features']
)

# Run for GRU actor-enriched model
fi_gru_a = permutation_importance_rnn(
    preds_final_gru_a['model'],
    preds_final_gru_a['X_test'],
    preds_final_gru_a['y_test_true'],
    preds_final_gru_a['y_test_scaled'],
    preds_final_gru_a['scaler_y'],
    preds_final_gru_a['features']
)

# Plot top 5 for LSTM
plt.figure(figsize=(8, 5))
fi_lstm_a.head(5).plot(kind='barh', color='green')
plt.title("Top 5 Features – LSTM Actor-Enriched (Permutation Importance)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Plot top 5 for GRU
plt.figure(figsize=(8, 5))
fi_gru_a.head(5).plot(kind='barh', color='red')
plt.title("Top 5 Features – GRU Actor-Enriched (Permutation Importance)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n Top Features – LSTM Actor-Enriched (Δ RMSE):\n")
print(fi_lstm_a.head(5).to_frame("Δ RMSE").to_markdown())

print("\n Top Features – GRU Actor-Enriched (Δ RMSE):\n")
print(fi_gru_a.head(5).to_frame("Δ RMSE").to_markdown())








