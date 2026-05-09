# 🌩️ WeatherGuard AI  
## AI-Based Weather Data Integrity System

WeatherGuard AI is an AI-powered weather data validation system that detects anomalies, sensor failures, tampering, and suspicious patterns in weather datasets using Machine Learning and Deep Learning techniques.

The system combines:
- Isolation Forest (Machine Learning)
- LSTM Autoencoder (Deep Learning)

to identify both point anomalies and sequential weather pattern abnormalities.

---

# 📌 Overview

Weather data plays a critical role in:
- Agriculture
- Aviation
- Disaster Management
- Climate Research
- Smart Cities

However, weather systems are vulnerable to:
- Sensor failures
- Missing or corrupted data
- Cyber attacks
- Data manipulation

WeatherGuard AI validates incoming weather data and detects suspicious activity automatically using AI models trained on historical climate data.

---

# ✨ Features

- AI-powered anomaly detection
- Real-time weather data validation
- Web dashboard for CSV uploads
- Visual anomaly reports and graphs
- Isolation Forest + LSTM hybrid detection
- Risk-based tampering verdict system
- Fast CPU-based execution

---

# 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Flask
- Pandas
- NumPy
- Matplotlib
- Joblib

---

# 🧪 Dataset

The project was trained using the **Jena Climate Dataset (2009–2016)** containing over 420,000 real weather sensor readings collected at 10-minute intervals.

Features include:
- Temperature
- Humidity
- Pressure
- Wind speed
- Dew point
- Vapor pressure

Dataset Source:
https://www.kaggle.com/datasets/mnassrib/jena-climate

---

# ⚙️ How It Works

# ⚙️ System Workflow

```text
Weather Dataset / API
        ↓
Data Preprocessing
        ↓
Isolation Forest Model
        ↓
LSTM Autoencoder
        ↓
Anomaly Detection
        ↓
Risk Analysis
        ↓
Dashboard + Reports
```

🤖 AI Models Used
1. Isolation Forest
Detects:
Outliers
Sudden spikes
Corrupted values
Point anomalies

2. LSTM Autoencoder
Detects:
Sequential anomalies
Pattern deviations
Subtle tampering
Sensor inconsistencies

The LSTM model learns normal weather behavior and flags abnormal sequences using reconstruction error thresholds.

# 🚀 Installation

Clone Repository- git clone https://github.com/SoorajSai/WeatherGuard-AI.git

cd WeatherGuard-AI

Create Virtual Environment

python -m venv venv

Activate (Windows)

venv\Scripts\activate

Activate (Mac/Linux)

source venv/bin/activate

Install Dependencies

pip install -r requirements.txt

▶️ Usage
Train Models
python train_model.py
Launch Web Application
python app.py

Open:http://127.0.0.1:5000

📊 Detection Verdicts
Tampered Percentage	Verdict
> 5%	🚨 HIGH RISK
1–5%	⚠️ MODERATE RISK
0.2–1%	🔶 LOW RISK
< 0.2%	✅ CLEAN

# 📁 Project Structure

```text
WeatherGuard-AI/
│
├── app.py
├── detect.py
├── train_model.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── plots/
│
├── weather_integrity_model/
│   ├── isolation_forest.pkl
│   ├── lstm_autoencoder.keras
│   ├── scaler.pkl
│   └── metadata.json
│
└── jena_climate_2009_2016.csv
```

📈 Output Features
The system generates:
Anomaly reports
Correlation graphs
Training loss graphs
Error distribution plots
Detection visualizations

🔮 Future Improvements
Real-time weather API integration
Blockchain-based data integrity
SMS/Email alerts
Cloud deployment
Mobile dashboard
Real-time streaming analysis

👨‍💻 Author
Sooraj Bangera
AIML/Cybersecurity Internship Project
