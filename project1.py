import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import zipfile
import os


zip_path = "Project10_Quality Prediction in a Mining Process.zip"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("mining_data")
    print("Dataset extracted successfully.")
    
    # Locate the CSV inside the extracted folder
    csv_file = [os.path.join("mining_data", f) for f in os.listdir("mining_data") if f.endswith('.csv')][0]
    df = pd.read_csv(csv_file)
else:
    print(f"Please ensure '{zip_path}' is in the working directory.")

    date_range = pd.date_range(start="2017-03-01", end="2017-09-01", freq="20S")
    df = pd.DataFrame(index=date_range).reset_index().rename(columns={'index': 'date'})
    df['% Iron Feed'] = np.random.uniform(50, 60, size=len(df))
    df['% Silica Feed'] = np.random.uniform(10, 15, size=len(df))
    df['Flotation Column 01 Air Flow'] = np.random.uniform(200, 300, size=len(df))
    df['Flotation Column 01 Level'] = np.random.uniform(150, 250, size=len(df))
    df['% Iron Concentrate'] = np.random.uniform(65, 68, size=len(df))
    df['% Silica Concentrate'] = 100 - df['% Iron Concentrate'] - np.random.uniform(1, 2, size=len(df))

print("Initial Data Shape:", df.shape)

# Convert date column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Replace comma decimals with dots if columns read as strings
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.').astype(float)

# Resample to 1-Minute Intervals (Answering Question 1: Minute-level Prediction)
# Process metrics are averaged; hourly lab metrics are forward-filled (ffill)
df_min = df.resample('1min').mean()
df_min['% Silica Concentrate'] = df_min['% Silica Concentrate'].ffill()
df_min['% Iron Concentrate'] = df_min['% Iron Concentrate'].ffill()
df_min.dropna(inplace=True)

print("Resampled 1-Minute Data Shape:", df_min.shape)

def create_features(data, target_col, drop_iron=False):
    X = data.copy()
    
    # Store target
    y = X[target_col]
    
    # Drop target columns to prevent leakage
    cols_to_drop = [target_col]
    if drop_iron and '% Iron Concentrate' in X.columns:
        cols_to_drop.append('% Iron Concentrate')
    elif not drop_iron and '% Iron Concentrate' in X.columns:
        cols_to_drop.append('% Iron Concentrate') # Exclude from X features if it's an end-of-pipe metric
        
    X = X.drop(columns=cols_to_drop)
    
    # Create 15-min, 30-min, and 60-min lag and rolling features for process variables
    process_cols = [c for c in X.columns if 'Air Flow' in c or 'Level' in c or 'Feed' in c]
    
    for col in process_cols:
        X[f'{col}_lag_15'] = X[col].shift(15)
        X[f'{col}_lag_60'] = X[col].shift(60)
        X[f'{col}_roll_mean_30'] = X[col].rolling(window=30).mean()
    
    X.dropna(inplace=True)
    y = y.loc[X.index]
    return X, y

print("\n--- Running Experiment: Predicting % Silica WITHOUT % Iron Concentrate ---")
X_exp1, y_exp1 = create_features(df_min, '% Silica Concentrate', drop_iron=True)

# Time Series Split (80% Train, 20% Test)
split_idx = int(len(X_exp1) * 0.8)
X_train, X_test = X_exp1.iloc[:split_idx], X_exp1.iloc[split_idx:]
y_train, y_test = y_exp1.iloc[:split_idx], y_exp1.iloc[split_idx:]

# Model Framework
model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions & Metrics
preds = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
print(f"R2 Score: {r2_score(y_test, preds):.4f}")


print("\n--- Running Experiment: Multi-Hour Ahead Forecasting Horizons ---")
horizons = [60, 120, 180] # 1 Hour, 2 Hours, 3 Hours ahead (in minutes)

for h in horizons:
    # Shift target backward to map current process state to future target state
    y_shifted = y_exp1.shift(-h)
    
    # Align X and shifted y
    valid_idx = y_shifted.dropna().index
    X_h = X_exp1.loc[valid_idx]
    y_h = y_shifted.loc[valid_idx]
    
    # Split
    split = int(len(X_h) * 0.8)
    X_tr, X_te = X_h.iloc[:split], X_h.iloc[split:]
    y_tr, y_te = y_h.iloc[:split], y_h.iloc[split:]
    
    # Train
    model_h = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_h.fit(X_tr, y_tr)
    
    # Eval
    p_h = model_h.predict(X_te)
    print(f"Horizon: {h//60} Hour(s) Ahead -> Test RMSE: {np.sqrt(mean_squared_error(y_te, p_h)):.4f} | R2: {r2_score(y_te, p_h):.4f}")

# Feature Importance Plot for the base model
feat_imp = pd.Series(model.feature_importances_, index=X_exp1.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title("Top 10 Crucial Process Variables Predicting % Silica")
plt.xlabel("Feature Importance Score")
plt.tight_layout()
plt.show()