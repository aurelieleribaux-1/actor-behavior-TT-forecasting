"""
ARIMA Residual Benchmark for Time Series Forecasting

This script:
- Loads train/test split data (`train.csv`, `test_final.csv`)
- Applies feature engineering to compute residual targets
- Trains an ARIMA(1,0,0) model on the residuals of TT
- Reconstructs the predicted TT by adding base values
- Evaluates RMSE and MAE
- Plots actual vs predicted TT over time

Dependencies:
- pandas, numpy, statsmodels, scikit-learn, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

from feature_engineering import feature_engineering

# ---------------------------
# Load data
# ---------------------------
df_train_full = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
df_test_final = pd.read_csv("data/test_final.csv", index_col=0, parse_dates=True)

# ---------------------------
#  Feature Engineering (Residual Target)
# ---------------------------
df_train_fe, _, _ = feature_engineering(df_train_full, residual_target=True)

df_context = pd.concat([df_train_full.tail(40), df_test_final])
df_test_fe, _, _ = feature_engineering(df_context, residual_target=True)
df_test_fe = df_test_fe.loc[df_test_final.index.intersection(df_test_fe.index)]

df_train_fe.dropna(inplace=True)
df_test_fe.dropna(inplace=True)

# ---------------------------
#  Extract residuals and base TT
# ---------------------------
base_values = df_test_fe['TT'].shift(1)
y_test_arima = df_test_fe['target']

df_eval = pd.DataFrame({
    'base': base_values,
    'target': y_test_arima
}).dropna().reset_index()

# Final aligned series
base_values = df_eval['base'].reset_index(drop=True)
y_test_arima = df_eval['target'].reset_index(drop=True)

# ---------------------------
# Train ARIMA on residuals
# ---------------------------
print(" Training ARIMA(0,1,0) on residuals...")
y_train_arima = df_train_fe['target']
model_arima = ARIMA(y_train_arima, order=(1, 0, 0))
model_arima_fit = model_arima.fit()

# Forecast residuals
resid_forecast = model_arima_fit.forecast(steps=len(y_test_arima))

# ---------------------------
# Reconstruct full TT and evaluate
# ---------------------------
min_len = min(len(resid_forecast), len(base_values), len(y_test_arima))

y_pred = resid_forecast[:min_len].reset_index(drop=True) + base_values[:min_len]
y_true = base_values[:min_len] + y_test_arima[:min_len]

rmse_arima = np.sqrt(mean_squared_error(y_true, y_pred))
mae_arima = mean_absolute_error(y_true, y_pred)

print(f"\n ARIMA Benchmark on Residuals → RMSE: {rmse_arima:.4f}, MAE: {mae_arima:.4f}")

# ---------------------------
#  Plot
# ---------------------------
plt.figure(figsize=(14, 6))
plt.plot(df_eval['index'][:min_len], y_true, label='Actual', color='black')
plt.plot(df_eval['index'][:min_len], y_pred, '--', label='ARIMA Residual Forecast', color='purple')
plt.title(" ARIMA Benchmark: Actual vs Reconstructed TT")
plt.xlabel("Time")
plt.ylabel("Duration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
