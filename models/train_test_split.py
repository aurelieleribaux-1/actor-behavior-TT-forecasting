# models/train_test_split.py

import pandas as pd

df = pd.read_csv("data/BPIC2017_timeseries.csv")  # or your chosen time series
df = df.sort_index()

n_test = int(len(df) * 0.20)
df_test_final = df.tail(n_test).copy()
df_train_full = df.drop(df_test_final.index).copy()

# Optionally save them for later reuse
df_train_full.to_csv("data/train.csv")
df_test_final.to_csv("data/test_final.csv")
