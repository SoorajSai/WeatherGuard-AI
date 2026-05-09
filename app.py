"""
Weather Data Integrity System — Flask Web App
Run: python app.py
Then open: http://127.0.0.1:5000
"""

import os, json, io, base64
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# ── Paths (adjust if needed) ──────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'weather_integrity_model')

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading models...")
iforest = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
    meta = json.load(f)

FEATURES  = meta['features']
SEQ_LEN   = meta['seq_len']
THRESHOLD = meta['threshold']

from tensorflow.keras.models import load_model as keras_load
autoencoder = keras_load(os.path.join(MODEL_DIR, 'lstm_autoencoder.keras'))
print("✓ Models loaded and ready.")


def make_chart(df_feat, if_anomaly, lstm_anomaly, recon_errors):
    """Generate a detection chart and return as base64 PNG."""
    combined = if_anomaly | lstm_anomaly
    n = len(df_feat)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), facecolor='#0d1117')
    fig.suptitle('Weather Data Integrity — Detection Report',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    plot_cfg = dict(facecolor='#161b22')
    spine_color = '#30363d'

    for ax in axes:
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.yaxis.label.set_color('#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')
        ax.grid(alpha=0.15, color='#30363d')

    # Panel 1: Temperature
    ax = axes[0]
    ax.plot(df_feat['T (degC)'], color='#58a6ff', lw=0.8, label='Temperature', zorder=2)
    if if_anomaly.any():
        ax.scatter(np.where(if_anomaly)[0], df_feat['T (degC)'].values[if_anomaly],
                   color='#f0883e', s=25, zorder=5, label='IF Anomaly', marker='o')
    if lstm_anomaly.any():
        ax.scatter(np.where(lstm_anomaly)[0], df_feat['T (degC)'].values[lstm_anomaly],
                   color='#ff4d4d', s=30, zorder=6, label='LSTM Anomaly', marker='x', lw=1.5)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature — Anomaly Overlay')
    ax.legend(loc='upper right', fontsize=8, facecolor='#21262d', labelcolor='white',
              edgecolor='#30363d')

    # Panel 2: Humidity
    ax = axes[1]
    colors = np.where(combined, '#ff4d4d', '#3fb950')
    ax.scatter(range(n), df_feat['rh (%)'], c=colors, s=2, alpha=0.7, zorder=2)
    ax.set_ylabel('Humidity (%)')
    ax.set_title('Humidity — Normal (green) vs Anomalous (red)')

    # Panel 3: Reconstruction error
    ax = axes[2]
    ax.plot(recon_errors, color='#a371f7', lw=0.7, label='Recon Error', zorder=2)
    ax.fill_between(range(n), recon_errors, 0,
                    where=recon_errors > THRESHOLD,
                    color='#ff4d4d', alpha=0.4, label='Above threshold')
    ax.axhline(THRESHOLD, color='#f85149', lw=1.5, linestyle='--',
               label=f'Threshold = {THRESHOLD:.5f}')
    ax.set_ylabel('MSE Error')
    ax.set_xlabel('Row Index')
    ax.set_title('LSTM Reconstruction Error')
    ax.legend(loc='upper right', fontsize=8, facecolor='#21262d', labelcolor='white',
              edgecolor='#30363d')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def run_detection(df_new):
    """Core detection logic. Returns a result dict."""
    missing = [c for c in FEATURES if c not in df_new.columns]
    if missing:
        return {'error': f'Missing columns: {missing}'}

    df_feat = df_new[FEATURES].copy().ffill().bfill().reset_index(drop=True)
    n_rows  = len(df_feat)

    if n_rows < 10:
        return {'error': 'Dataset too small (need at least 10 rows)'}

    X = scaler.transform(df_feat)

    # ── Isolation Forest ──────────────────────────────────────────────────────
    if_scores = iforest.decision_function(X)
    if_labels = iforest.predict(X)
    if_anomaly = (if_labels == -1)
    n_if = int(if_anomaly.sum())

    # ── LSTM Autoencoder ──────────────────────────────────────────────────────
    lstm_anomaly  = np.zeros(n_rows, dtype=bool)
    recon_errors  = np.zeros(n_rows)

    if n_rows >= SEQ_LEN + 10:
        seqs, indices = [], []
        for i in range(n_rows - SEQ_LEN):
            seqs.append(X[i:i + SEQ_LEN])
            indices.append(i + SEQ_LEN - 1)
        seqs = np.array(seqs, dtype=np.float32)
        preds  = autoencoder.predict(seqs, batch_size=256, verbose=0)
        errors = np.mean(np.mean(np.square(seqs - preds), axis=2), axis=1)
        for idx, err in zip(indices, errors):
            recon_errors[idx] = float(err)
            if err > THRESHOLD:
                lstm_anomaly[idx] = True
        n_lstm = int(lstm_anomaly.sum())
        lstm_used = True
    else:
        n_lstm    = 0
        lstm_used = False

    # ── Combined verdict ──────────────────────────────────────────────────────
    combined    = if_anomaly | lstm_anomaly
    n_combined  = int(combined.sum())
    tamper_pct  = round(100 * n_combined / n_rows, 2)

    if tamper_pct > 5:
        verdict = 'HIGH_RISK'
        verdict_text = '🚨 HIGH RISK — Dataset appears TAMPERED'
        color = '#ff4d4d'
    elif tamper_pct > 1:
        verdict = 'MODERATE'
        verdict_text = '⚠️ MODERATE RISK — Suspicious anomalies detected'
        color = '#f0883e'
    elif tamper_pct > 0.2:
        verdict = 'LOW_RISK'
        verdict_text = '🔶 LOW RISK — Minor irregularities found'
        color = '#d29922'
    else:
        verdict = 'CLEAN'
        verdict_text = '✅ CLEAN — Dataset appears AUTHENTIC'
        color = '#3fb950'

    # ── Per-feature breakdown ─────────────────────────────────────────────────
    feature_stats = []
    for feat in FEATURES:
        col = df_feat[feat].values
        anom_mask = combined
        feature_stats.append({
            'name':       feat,
            'normal_min': round(float(col.min()), 3),
            'normal_max': round(float(col.max()), 3),
            'normal_mean':round(float(col.mean()), 3),
            'anom_count': int(anom_mask.sum()),
            'anom_min':   round(float(col[anom_mask].min()), 3) if anom_mask.any() else None,
            'anom_max':   round(float(col[anom_mask].max()), 3) if anom_mask.any() else None,
        })

    # ── Chart ─────────────────────────────────────────────────────────────────
    chart_b64 = make_chart(df_feat, if_anomaly, lstm_anomaly, recon_errors)

    return {
        'verdict':        verdict,
        'verdict_text':   verdict_text,
        'color':          color,
        'total_rows':     n_rows,
        'anomaly_rows':   n_combined,
        'tamper_pct':     tamper_pct,
        'if_anomalies':   n_if,
        'lstm_anomalies': n_lstm,
        'lstm_used':      lstm_used,
        'threshold':      round(THRESHOLD, 6),
        'feature_stats':  feature_stats,
        'chart':          chart_b64,
    }


@app.route('/')
def index():
    return render_template('index.html', features=FEATURES)


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a .csv file'}), 400
    try:
        df = pd.read_csv(f)
        result = run_detection(df)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n🌤️  Weather Integrity System — starting server...")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)