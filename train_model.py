"""
AI-Based Weather Data Integrity System
Train Script - Jena Climate Dataset (2009-2016)
Uses: Isolation Forest (fast ML) + LSTM Autoencoder (deep learning)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH  = "jena_climate_2009_2016.csv"
MODEL_DIR  = "weather_integrity_model"
PLOTS_DIR  = "plots"
SEQ_LEN      = 144          # 24 hours (10-min intervals)
EPOCHS       = 20
BATCH_SIZE   = 64
SAMPLE_FRAC  = 0.3          # use 30% of data for speed (still ~120k rows)
CONTAMINATION = 0.03        # 3% of data assumed anomalous

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES = ['p (mbar)', 'T (degC)', 'Tdew (degC)', 'rh (%)',
            'VPmax (mbar)', 'VPact (mbar)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']

# ─── 1. LOAD & CLEAN DATA ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and cleaning Jena Climate dataset...")

df = pd.read_csv(DATA_PATH, parse_dates=['Date Time'])
df = df.sort_values('Date Time').reset_index(drop=True)

# Remove known bad rows (wv = -9999)
df = df[df['wv (m/s)'] >= 0]
df = df[df['max. wv (m/s)'] >= 0]

# Sample for tractable training
df = df.sample(frac=SAMPLE_FRAC, random_state=42).sort_values('Date Time').reset_index(drop=True)

print(f"  Dataset shape: {df.shape}")
print(f"  Date range: {df['Date Time'].min()} → {df['Date Time'].max()}")
print(f"  Features used: {FEATURES}")

# ─── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\nSTEP 2: Feature engineering...")

data = df[FEATURES].copy()
data = data.ffill().bfill()   # fill any gaps

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

print(f"  Scaled shape: {data_scaled.shape}")

# ─── 3. ISOLATION FOREST (ML Model) ──────────────────────────────────────────
print("\nSTEP 3: Training Isolation Forest model...")

iforest = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1
)
iforest.fit(data_scaled)

if_scores = iforest.decision_function(data_scaled)
if_labels = iforest.predict(data_scaled)   # -1 = anomaly, 1 = normal

n_anomalies_if = (if_labels == -1).sum()
print(f"  Isolation Forest anomalies found: {n_anomalies_if} ({100*n_anomalies_if/len(if_labels):.2f}%)")

# Save IF model + scaler
joblib.dump(iforest, f"{MODEL_DIR}/isolation_forest.pkl")
joblib.dump(scaler,  f"{MODEL_DIR}/scaler.pkl")
print("  ✓ Isolation Forest model saved")

# ─── 4. BUILD SEQUENCES FOR LSTM AUTOENCODER ──────────────────────────────────
print("\nSTEP 4: Building time-series sequences for LSTM autoencoder...")

def build_sequences(data, seq_len):
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
    return np.array(X)

X_seq = build_sequences(data_scaled, SEQ_LEN)
print(f"  Sequence shape: {X_seq.shape}  (samples × timesteps × features)")

# Train/val split (80/20)
split = int(len(X_seq) * 0.8)
X_train = X_seq[:split]
X_val   = X_seq[split:]
print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}")

# ─── 5. LSTM AUTOENCODER ──────────────────────────────────────────────────────
print("\nSTEP 5: Building and training LSTM Autoencoder...")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, RepeatVector,
                                     TimeDistributed, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

n_features = len(FEATURES)

inputs = Input(shape=(SEQ_LEN, n_features))
# Encoder
x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(0.2)(x)
x = LSTM(32, return_sequences=False)(x)
encoded = Dense(16, activation='relu')(x)
# Decoder
x = RepeatVector(SEQ_LEN)(encoded)
x = LSTM(32, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(64, return_sequences=True)(x)
decoded = TimeDistributed(Dense(n_features))(x)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)
]

history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ─── 6. COMPUTE RECONSTRUCTION THRESHOLD ─────────────────────────────────────
print("\nSTEP 6: Computing anomaly threshold from validation set...")

X_val_pred = autoencoder.predict(X_val, verbose=0)
val_errors = np.mean(np.mean(np.square(X_val - X_val_pred), axis=2), axis=1)

# Threshold = mean + 3 std (captures rare events)
threshold = float(np.mean(val_errors) + 3 * np.std(val_errors))
print(f"  Reconstruction error — mean: {val_errors.mean():.6f}, std: {val_errors.std():.6f}")
print(f"  Anomaly threshold (μ + 3σ): {threshold:.6f}")

# ─── 7. SAVE MODELS & METADATA ────────────────────────────────────────────────
print("\nSTEP 7: Saving models and metadata...")

autoencoder.save(f"{MODEL_DIR}/lstm_autoencoder.keras")

metadata = {
    "features": FEATURES,
    "seq_len": SEQ_LEN,
    "threshold": threshold,
    "contamination": CONTAMINATION,
    "val_error_mean": float(val_errors.mean()),
    "val_error_std": float(val_errors.std()),
    "n_train_samples": int(X_train.shape[0]),
    "n_val_samples": int(X_val.shape[0]),
    "final_train_loss": float(history.history['loss'][-1]),
    "final_val_loss": float(history.history['val_loss'][-1]),
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ LSTM Autoencoder saved → {MODEL_DIR}/lstm_autoencoder.keras")
print(f"  ✓ Metadata saved         → {MODEL_DIR}/metadata.json")

# ─── 8. PLOTS ─────────────────────────────────────────────────────────────────
print("\nSTEP 8: Generating diagnostic plots...")

# Plot 1: Training loss
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'],     label='Train Loss', color='royalblue')
ax.plot(history.history['val_loss'], label='Val Loss',   color='tomato')
ax.set_title('LSTM Autoencoder — Training Loss', fontsize=14)
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/training_loss.png", dpi=120)
plt.close()

# Plot 2: Reconstruction error distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(val_errors, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(threshold, color='red', linewidth=2, label=f'Threshold = {threshold:.5f}')
ax.set_title('Validation Reconstruction Error Distribution', fontsize=14)
ax.set_xlabel('Mean Squared Reconstruction Error')
ax.set_ylabel('Count')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/error_distribution.png", dpi=120)
plt.close()

# Plot 3: Isolation Forest anomalies on temperature
fig, ax = plt.subplots(figsize=(14, 4))
colors = np.where(if_labels == -1, 'red', 'steelblue')
ax.scatter(df.index, df['T (degC)'], c=colors, s=1, alpha=0.5)
ax.set_title('Isolation Forest — Temperature Anomalies (red)', fontsize=14)
ax.set_xlabel('Row Index'); ax.set_ylabel('Temperature (°C)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/isolation_forest_anomalies.png", dpi=120)
plt.close()

# Plot 4: Feature correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[FEATURES].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_correlation.png", dpi=120)
plt.close()

print(f"  ✓ Plots saved to {PLOTS_DIR}/")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"  Models saved in: {MODEL_DIR}/")
print(f"  Files:")
print(f"    • isolation_forest.pkl  - ML anomaly detector")
print(f"    • lstm_autoencoder.keras - Deep learning model")
print(f"    • scaler.pkl            - Feature normalizer")
print(f"    • metadata.json         - Threshold & config")
print(f"  Plots saved in: {PLOTS_DIR}/")
print("  Run detect.py to check any new dataset for tampering.")