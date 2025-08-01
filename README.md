#  Actor Behavior Forecasting using Multivariate Time Series

This repository provides a modular pipeline to forecast process performance using actor behavior extracted from event logs. It supports classical models (XGBoost, LightGBM, ARIMA) and deep learning (RNNs with attention).

---

## Project Structure
actor-behavior-tt_forecasting/
│
├── README.md
├── requirements.txt
│
├── dataset pipeline/
│ ├── time_series_generation.py # Event log + actor behavior → final time series
│
├── models/
│ ├── xgboost_model.py # XGBoost (baseline & actor-enriched)
│ ├── lightgbm_model.py # LightGBM (baseline & actor-enriched)
│ ├── arima_model.py # ARIMA residual forecasting
│ ├── RNN_Attn.py # GRU/LSTM with attention
│ ├──RNN_model.py # GRU/LSTM without attention
│ ├──feature_engineering.py
│
├── utils/
│ └── data_utils.py # Load and split time series
│ ├──train_test_split.py
└── data/
├── final_multivariatetimeseries.csv

## Usage Overview

### 1️ Generate Multivariate Time Series

Before training any models, you must generate the final multivariate time series, which includes:

- Throughput time (TT)
- Actor behavior frequencies and durations
- Resource participation

First, place your input files in the data/ folder:
1.The original event log (your_file.xes)

2.The actor behavior file (your_file.csv) generated via: [Linking Actor Behavior to Process Performance Over Time](https://arxiv.org/abs/2507.23037)

The, run the following script:

python dataset pipeline/time_series_generation.py

This will automatically generate the multivariate time series: data/final_multivariatetimeseries.csv

All model scripts automatically load the final time series and split it using a shared utility:

from utils.data_utils import load_train_test_split

df_train_full, df_test_final = load_train_test_split("data/final_multivariatetimeseries.csv")

### Train and Evaluate Forecasting Models 

Each model script follows the same structure:

1. Loads and splits the multivariate time series

2. Applies baseline vs actor-enriched feature engineering

3. Trains and validates the model using cross-validation

4. Evaluates on a final holdout test set

5. Visualizes predictions and feature importances

Available Models:
python models/xgboost_model.py
python models/lightgbm_model.py
python models/rnn_with_attention.py
python models/arima_model.py

### Example Output : 
Cross-validation RMSE/MAE/R^2 (with confidence intervals)

Holdout evaluation (RMSE, MAE, R²)

SHAP or permutation feature importance

Actual vs predicted time series plots

### Requirements 
Install with:

pip install -r requirements.txt

