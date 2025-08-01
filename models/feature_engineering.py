#feature_engineering.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def feature_engineering(df, tt_lags=range(1, 4), actor_lags=range(1, 21), rolling_windows=[3, 7, 14], residual_target=True):
    df_model = df.copy()

    while df_model['TT'].iloc[-1] == 0:
        df_model = df_model.iloc[:-1]

    if residual_target:
        df_model['target'] = df_model['TT'].diff().rolling(window=3).mean().shift(1)
    else:
        df_model['target'] =df_model['TT'].shift(-1)

    # TT lags
    for lag in tt_lags:
        df_model[f'TT_lag{lag}'] = df_model['TT'].shift(lag)

    # TT rolling stats
    for window in rolling_windows:
        df_model[f'TT_rolling_mean{window}'] = df_model['TT'].rolling(window).mean().shift(1)
        df_model[f'TT_rolling_std{window}'] = df_model['TT'].rolling(window).std().shift(1)
        df_model[f'TT_rolling_max{window}'] = df_model['TT'].rolling(window).max().shift(1)

    df_model['TT_zscore7'] = (
        (df_model['TT'].shift(1) - df_model['TT_rolling_mean7']) /
        df_model['TT_rolling_std7']
    )

    actor_vars = ['Count_C', 'Count_HB', 'Count_I', 'Count_HI',
                  'Time_C_seconds', 'Time_I_seconds', 'Time_HI_seconds',
                  'Time_HB_seconds']

    for var in actor_vars:
        if var in df_model.columns:
            # Actor lags
            for lag in actor_lags:
                df_model[f'{var}_lag{lag}'] = df_model[var].shift(lag)

            # Rolling stats
            for window in rolling_windows:
                df_model[f'{var}_rolling_mean{window}'] = df_model[var].rolling(window).mean().shift(1)
                df_model[f'{var}_rolling_std{window}'] = df_model[var].rolling(window).std().shift(1)
                df_model[f'{var}_rolling_max{window}'] = df_model[var].rolling(window).max().shift(1)

            # Z-score
            df_model[f'{var}_zscore7'] = (
                (df_model[var].shift(1) - df_model[f'{var}_rolling_mean7']) /
                df_model[f'{var}_rolling_std7']
            )

    # Peak indicator
    peaks, _ = find_peaks(df_model['TT'], distance=7)
    df_model['peak_flag'] = 0
    df_model.iloc[peaks, df_model.columns.get_loc('peak_flag')] = 1

    df_model.dropna(inplace=True)

    # Feature lists
    baseline_features = [f'TT_lag{lag}' for lag in tt_lags]
    for w in rolling_windows:
        baseline_features += [f'TT_rolling_mean{w}', f'TT_rolling_std{w}', f'TT_rolling_max{w}']
    baseline_features += ['TT_zscore7', 'peak_flag']

    actor_features = []
    for var in actor_vars:
        if var in df_model.columns:
            actor_features.append(var)
            actor_features += [f'{var}_lag{lag}' for lag in actor_lags]

            for window in rolling_windows:
                for stat in ['rolling_mean', 'rolling_std', 'rolling_max']:
                    actor_features.append(f'{var}_{stat}{window}')

            actor_features.append(f'{var}_zscore7')

    actor_features = [f for f in actor_features if f not in baseline_features]

    return df_model.copy(), baseline_features, actor_features

