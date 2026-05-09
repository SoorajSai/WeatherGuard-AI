"""
AI-Based Weather Data Integrity System
Detect Script — run this on any new weather CSV to check for tampering.

Usage:
    python detect.py <path_to_new_weather_csv>

The CSV must contain these columns (same as Jena Climate format):
    p (mbar), T (degC), Tdew (degC), rh (%), VPmax (mbar),
    VPact (mbar), wv (m/s), max. wv (m/s), wd (deg)
"""

import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = "weather_integrity_model"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
print("Loading models...")
iforest     = joblib.load(f"{MODEL_DIR}/isolation_forest.pkl")
scaler      = joblib.load(f"{MODEL_DIR}/scaler.pkl")

with open(f"{MODEL_DIR}/metadata.json") as f:
    meta = json.load(f)

FEATURES   = meta["features"]
SEQ_LEN    = meta["seq_len"]
THRESHOLD  = meta["threshold"]

from tensorflow.keras.models import load_model
autoencoder = load_model(f"{MODEL_DIR}/lstm_autoencoder.keras")
print(f"  ✓ All models loaded | Anomaly threshold: {THRESHOLD:.6f}")

# ─── LOAD NEW DATASET ─────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    # Demo: create a synthetic tampered dataset for testing
    print("\nNo CSV provided. Running DEMO with synthetic tampered data...\n")
    np.random.seed(0)
    n = 2000
    t = np.linspace(0, 8*np.pi, n)
    demo = pd.DataFrame({
        'p (mbar)':       990 + 10*np.sin(t) + np.random.normal(0, 0.5, n),
        'T (degC)':       10  + 8*np.sin(t/2) + np.random.normal(0, 0.3, n),
        'Tdew (degC)':    5   + 6*np.sin(t/2) + np.random.normal(0, 0.3, n),
        'rh (%)':         70  + 15*np.sin(t)  + np.random.normal(0, 1, n),
        'VPmax (mbar)':   12  + 4*np.sin(t/2) + np.random.normal(0, 0.2, n),
        'VPact (mbar)':   9   + 3*np.sin(t/2) + np.random.normal(0, 0.2, n),
        'wv (m/s)':       2   + np.abs(np.random.normal(0, 1, n)),
        'max. wv (m/s)':  3   + np.abs(np.random.normal(0, 1.5, n)),
        'wd (deg)':       180 + 90*np.sin(t)  + np.random.normal(0, 5, n),
    })
    # Inject tampered values (spikes + flat-lines)
    tamper_idx = np.random.choice(n, size=60, replace=False)
    demo.loc[tamper_idx[:30], 'T (degC)'] += np.random.uniform(25, 50, 30)   # temp spikes
    demo.loc[tamper_idx[30:], 'rh (%)']    = 999.0                            # sensor stuck
    csv_path = r"C:\Users\ASUS\Documents\!WORKSAPCE\!Internship_Mysuru\weather_data.csv"
    demo.to_csv(csv_path, index=False)
    print(f"  Demo dataset saved: {csv_path}  ({n} rows, ~90 tampered)")
else:
    csv_path = sys.argv[1]

print(f"\nLoading dataset: {csv_path}")
df_new = pd.read_csv(csv_path)

# Check required columns
missing = [c for c in FEATURES if c not in df_new.columns]
if missing:
    print(f"\n[ERROR] Missing columns in your CSV: {missing}")
    print(f"  Required columns: {FEATURES}")
    sys.exit(1)

df_feat = df_new[FEATURES].copy().ffill().bfill().reset_index(drop=True)
n_rows = len(df_feat)
print(f"  Rows: {n_rows}  |  Features: {len(FEATURES)}")

if n_rows < SEQ_LEN + 10:
    print(f"\n[WARNING] Dataset too small for LSTM (need >{SEQ_LEN} rows). "
          f"Will use Isolation Forest only.")
    use_lstm = False
else:
    use_lstm = True

# ─── SCALE ────────────────────────────────────────────────────────────────────
X = scaler.transform(df_feat)

# ─── ISOLATION FOREST DETECTION ───────────────────────────────────────────────
print("\n── Isolation Forest Detection ──")
if_scores  = iforest.decision_function(X)
if_labels  = iforest.predict(X)           # -1 anomaly
if_anomaly = (if_labels == -1)
n_if = if_anomaly.sum()
print(f"  Anomalies detected: {n_if} / {n_rows}  ({100*n_if/n_rows:.2f}%)")

# ─── LSTM AUTOENCODER DETECTION ───────────────────────────────────────────────
lstm_anomaly = np.zeros(n_rows, dtype=bool)
recon_errors = np.zeros(n_rows)

if use_lstm:
    print("\n── LSTM Autoencoder Detection ──")
    seqs, indices = [], []
    for i in range(n_rows - SEQ_LEN):
        seqs.append(X[i:i + SEQ_LEN])
        indices.append(i + SEQ_LEN - 1)   # flag the LAST point in the window
    seqs = np.array(seqs)

    preds  = autoencoder.predict(seqs, batch_size=64, verbose=0)
    errors = np.mean(np.mean(np.square(seqs - preds), axis=2), axis=1)

    for idx, err in zip(indices, errors):
        recon_errors[idx] = err
        if err > THRESHOLD:
            lstm_anomaly[idx] = True

    n_lstm = lstm_anomaly.sum()
    print(f"  Threshold:          {THRESHOLD:.6f}")
    print(f"  Max recon error:    {errors.max():.6f}")
    print(f"  Mean recon error:   {errors.mean():.6f}")
    print(f"  Anomalies detected: {n_lstm} / {n_rows}  ({100*n_lstm/n_rows:.2f}%)")

# ─── COMBINED VERDICT ─────────────────────────────────────────────────────────
combined = if_anomaly | lstm_anomaly
n_combined = combined.sum()

print("\n" + "=" * 55)
print("  TAMPERING VERDICT")
print("=" * 55)
tamper_pct = 100 * n_combined / n_rows

if tamper_pct > 5:
    verdict = "🚨  HIGH RISK — Dataset appears TAMPERED"
elif tamper_pct > 1:
    verdict = "⚠️   MODERATE RISK — Suspicious anomalies detected"
elif tamper_pct > 0.2:
    verdict = "🔶  LOW RISK — Minor irregularities found"
else:
    verdict = "✅  CLEAN — Dataset appears AUTHENTIC"

print(f"\n  {verdict}")
print(f"\n  Total anomalous rows : {n_combined} / {n_rows}")
print(f"  Tampered percentage  : {tamper_pct:.2f}%")
print(f"  Isolation Forest     : {n_if} anomalies")
if use_lstm:
    print(f"  LSTM Autoencoder     : {n_lstm} anomalies")

# Per-feature anomaly breakdown
print("\n  Feature-wise anomaly breakdown:")
for feat in FEATURES:
    col_vals = df_feat[feat].values
    anom_vals = col_vals[combined]
    if len(anom_vals) > 0:
        print(f"    {feat:<22} anomaly range: [{anom_vals.min():.2f}, {anom_vals.max():.2f}]"
              f"  normal: [{col_vals.min():.2f}, {col_vals.max():.2f}]")

# ─── PLOTS ────────────────────────────────────────────────────────────────────
print("\nGenerating report plots...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle(f"Weather Data Integrity Report\n{os.path.basename(csv_path)}", fontsize=14, fontweight='bold')

# Panel 1: Temperature with anomalies highlighted
ax = axes[0]
ax.plot(df_feat['T (degC)'], color='steelblue', linewidth=0.8, label='Temperature')
ax.scatter(np.where(if_anomaly)[0], df_feat['T (degC)'].values[if_anomaly],
           color='orange', s=20, zorder=5, label='IF Anomaly')
if use_lstm:
    ax.scatter(np.where(lstm_anomaly)[0], df_feat['T (degC)'].values[lstm_anomaly],
               color='red', s=20, zorder=5, label='LSTM Anomaly', marker='x')
ax.set_ylabel('Temperature (°C)'); ax.set_title('Temperature Anomaly Detection')
ax.legend(loc='upper right'); ax.grid(alpha=0.3)

# Panel 2: Humidity with anomalies
ax = axes[1]
ax.plot(df_feat['rh (%)'], color='teal', linewidth=0.8, label='Humidity')
ax.scatter(np.where(combined)[0], df_feat['rh (%)'].values[combined],
           color='red', s=20, zorder=5, label='Anomaly (combined)', marker='x')
ax.set_ylabel('Relative Humidity (%)'); ax.set_title('Humidity Anomaly Detection')
ax.legend(loc='upper right'); ax.grid(alpha=0.3)

# Panel 3: Reconstruction error (if LSTM used)
ax = axes[2]
if use_lstm:
    ax.plot(recon_errors, color='purple', linewidth=0.6, label='Recon Error')
    ax.axhline(THRESHOLD, color='red', linewidth=1.5, linestyle='--', label=f'Threshold={THRESHOLD:.5f}')
    ax.fill_between(range(n_rows), recon_errors, THRESHOLD,
                    where=recon_errors > THRESHOLD, color='red', alpha=0.3)
    ax.set_ylabel('MSE Reconstruction Error'); ax.set_title('LSTM Autoencoder Reconstruction Error')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)
else:
    ax.bar(range(n_rows), -if_scores, color=np.where(if_anomaly, 'red', 'steelblue'), width=1)
    ax.set_ylabel('Anomaly Score'); ax.set_title('Isolation Forest Anomaly Scores')
    ax.grid(alpha=0.3)

plt.tight_layout()
out_plot = f"{PLOTS_DIR}/detection_report.png"
plt.savefig(out_plot, dpi=130)
plt.close()
print(f"  ✓ Detection report saved: {out_plot}")

# Save anomaly rows
anomaly_df = df_new[combined].copy()
anomaly_df['anomaly_score_IF'] = -if_scores[combined]
out_csv = f"{PLOTS_DIR}/anomaly_rows.csv"
anomaly_df.to_csv(out_csv, index=False)
print(f"  ✓ Anomaly rows saved: {out_csv}  ({len(anomaly_df)} rows)")

print("\nDone.")