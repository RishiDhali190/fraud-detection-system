# 🛡️ End-to-End Fraud Detection System

A **production-style** machine-learning project that demonstrates an end-to-end fraud detection pipeline — from raw data to a live Streamlit demo.

> Built for portfolio / GitHub showcase purposes.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Data Cleaning Pipeline** | Handles missing values (median imputation), scaling (StandardScaler), type enforcement |
| **Imbalanced Data Handling** | SMOTE over-sampling + `class_weight="balanced"` for supported models |
| **Model Comparison** | Logistic Regression vs Random Forest vs Gradient Boosting |
| **Evaluation Beyond Accuracy** | Precision, Recall, F1-Score, ROC-AUC, **PR-AUC** (best metric for imbalanced data) |
| **Streamlit Demo** | Interactive dashboard with data explorer, model comparison, and real-time prediction |

---

## 📁 Project Structure

```
fraud_detection_system/
├── app.py                  # Streamlit demo application
├── run_pipeline.py         # One-command ML pipeline orchestrator
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py    # Synthetic imbalanced dataset generator
│   ├── data_pipeline.py    # Loading, cleaning, preprocessing
│   ├── modeling.py         # SMOTE + model training
│   └── evaluation.py       # Metrics, comparison, plotting
│
├── data/                   # Generated CSV (git-ignored)
│   └── fraud_data.csv
│
└── models/                 # Saved artifacts (git-ignored)
    ├── best_model.pkl
    ├── best_model_name.pkl
    ├── preprocessor.pkl
    └── feature_names.pkl
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fraud_detection_system.git
cd fraud_detection_system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

This command will:
- ✅ Generate a synthetic fraud dataset (10,000 transactions, ~1.7 % fraud)
- ✅ Clean and preprocess the data
- ✅ Apply SMOTE to handle class imbalance
- ✅ Train Logistic Regression, Random Forest, and Gradient Boosting
- ✅ Evaluate and compare models (PR-AUC, F1, ROC-AUC …)
- ✅ Save the best model to `models/`

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Streamlit App Tabs

| Tab | What it shows |
|---|---|
| **📊 Data Explorer** | Dataset stats, class distribution, transaction amounts, hourly patterns, missing-value report |
| **🏆 Model Results** | Side-by-side metrics table, ROC curves, Precision-Recall curves, confusion matrices, training times |
| **🔮 Live Prediction** | Enter transaction features (or load a random sample) and get an instant fraud probability |

---

## 🧠 Technical Details

### Why PR-AUC over ROC-AUC?

With heavily imbalanced data (~1.7 % fraud), ROC-AUC can be misleadingly optimistic because it is dominated by the large number of true negatives. **Precision-Recall AUC** focuses on the minority (fraud) class performance and is the primary ranking metric in this project.

### SMOTE

Synthetic Minority Over-sampling Technique generates synthetic fraud samples by interpolating between existing minority instances, giving classifiers more signal to learn from.

### Class Weights

Models that support `class_weight="balanced"` (Logistic Regression, Random Forest) automatically up-weight the minority class during training, complementing SMOTE.

---

## 📜 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use it for learning, portfolios, or as a starting point for production systems.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

*Built with ❤️ using scikit-learn, imbalanced-learn, and Streamlit.*
