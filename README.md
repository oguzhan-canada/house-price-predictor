# 🏠 House Price Predictor — Azure AI Foundry + Phi-4

An end-to-end ML project that predicts house sale prices using a Random Forest model, with AI-powered interpretation via Microsoft Phi-4 deployed on Azure AI Foundry.

## 🔗 Live Demo
[👉 Try the Interactive App Here](https://oguzhantekin.github.io/house-price-predictor)

---

## 📌 Project Overview

This project demonstrates a complete MLOps pipeline combining traditional machine learning with a large language model (LLM) to not only predict house prices but explain those predictions in plain English — just like a real estate analyst would. It present results in a strucutred format for audiences. 

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core programming language |
| Scikit-learn | Random Forest regression model |
| Pandas / NumPy | Data cleaning and feature engineering |
| MLflow | Experiment tracking and model logging |
| Azure AI Foundry | LLM deployment platform |
| Microsoft Phi-4 | AI interpretation of predictions |
| Joblib | Model serialization |
| HTML / CSS / JavaScript | Interactive front-end web application |

---

## 📁 Project Structure

```
dataset-hpp/
├── train.csv                  # Kaggle training dataset (1460 houses, 81 features)
├── test.csv                   # Kaggle test dataset
├── data_description.txt       # Feature descriptions
├── sample_submission.csv      # Kaggle submission format
├── eda.py                     # Exploratory Data Analysis + data cleaning
├── train.py                   # Model training + MLflow logging
├── interpret.py               # Phi-4 prediction interpretation
├── pipeline.py                # Full end-to-end pipeline
├── house_price_model.pkl      # Saved trained model
├── index.html                 # Interactive front-end web application
├── mlruns/                    # MLflow experiment tracking data
└── README.md                  # This file
```

---

## 📊 Dataset

**Source:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

- **1,460** training samples
- **81** features (structural, quality, location, and sale attributes)
- **Target:** `SalePrice` (continuous — regression problem)
- **Price range:** $34,900 — $755,000 (mean: $180,921)

---

## 🔧 Data Preprocessing

- Filled categorical missing values (e.g., `PoolQC`, `Alley`) with `"None"` — representing absence, not unknown
- Filled numeric missing values (e.g., `GarageArea`, `MasVnrArea`) with `0`
- Filled `LotFrontage` with **neighborhood median** — a smarter fill than global median
- Filled `Electrical` with the most common value (mode)

---

## ⚙️ Feature Engineering

Four new features were created to improve model performance:

| Feature | Formula | Rationale |
|---|---|---|
| `TotalSF` | Basement + 1st Floor + 2nd Floor sqft | Overall size is a top price driver |
| `TotalBath` | Full baths + 0.5 × half baths | Bathroom count affects value |
| `HouseAge` | Year Sold − Year Built | Older homes typically sell for less |
| `Remodeled` | 1 if remodeled, else 0 | Remodeling adds value |

---

## 🤖 Model Performance

**Algorithm:** Random Forest Regressor (100 trees)

| Metric | Value | Meaning |
|---|---|---|
| R² | 0.887 | Model explains 88.7% of price variance |
| MAE | $17,785 | Average prediction error |
| RMSE | $29,445 | Penalizes larger errors more |

All runs tracked and logged via **MLflow**.

---

## 🧠 Phi-4 Interpretation (Sample Output)

```
House Details:
- Overall Quality: 6/10
- Total Square Footage: 2,127 sqft
- House Age: 43 years
- Bathrooms: 1.5
- Garage: 1 car
- Remodeled: Yes

Predicted Price: $141,130
Actual Price:    $154,500

Phi-4 Says:
"The predicted price reflects the average quality rating and moderate square 
footage. The 43-year age contributes to a lower valuation despite remodeling. 
The single-car garage and 1.5 bathrooms are standard features that support 
this price point."
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
python -m pip install pandas numpy scikit-learn mlflow openai joblib
```

### 2. Run EDA and data cleaning
```bash
python eda.py
```

### 3. Train the model
```bash
python train.py
```

### 4. View MLflow experiments
```bash
python -m mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri "file:///path/to/mlruns"
```

### 5. Run full pipeline with Phi-4 interpretation
```bash
python pipeline.py
```

---

## ☁️ Azure AI Foundry Setup

1. Create a project at [ai.azure.com](https://ai.azure.com)
2. Deploy **Phi-4** from the Model Catalog as a Serverless API
3. Copy your endpoint URL and API key
4. Add credentials to `pipeline.py`

**Estimated cost for this project:** < $3 





