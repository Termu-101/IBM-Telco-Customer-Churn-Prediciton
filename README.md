# 🔄 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-316192?style=for-the-badge&logo=postgresql)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-ML-F7931E?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

---

## 📌 Business Problem

A telecom company is losing customers every month with no way to identify who will leave before they do. This project builds an end-to-end machine learning system that predicts which customers are at risk of churning — enabling the business to take proactive retention action before it's too late.

> **Key Business Question:** Which customers are most likely to cancel their subscription in the next 30 days?

---

## 🏗️ Project Architecture
```
Raw Data (CSV)
     ↓
Python (Cleaning)
     ↓
PostgreSQL (Storage + Feature Engineering)
     ↓
Python (ML Pipeline — XGBoost)
     ↓
FastAPI (REST API — Real Time Predictions)
```

---

## 📊 Results

| Model | ROC-AUC | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.84 | 0.63 | 0.78 | 0.70 |
| Random Forest | 0.87 | 0.69 | 0.76 | 0.72 |
| **XGBoost** | **0.91** | **0.72** | **0.79** | **0.75** |

✅ XGBoost selected as final model with **0.91 ROC-AUC**

---

## 🗂️ Project Structure
```
customer-churn-prediction/
├── notebooks/
│   └── churn_analysis.ipynb       # Full ML pipeline
├── sql/
│   ├── create_tables.sql          # Table creation
│   ├── exploratory_analysis.sql   # 15 analysis queries
│   └── ml_features_view.sql       # Feature engineering view
├── api/
│   └── app.py                     # FastAPI endpoint
├── data/
│   └── telco_churn.csv            # Raw dataset
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Storage | PostgreSQL |
| Data Connection | SQLAlchemy |
| Analysis | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Explainability | SHAP |
| API | FastAPI, Uvicorn |
| Version Control | Git, GitHub |

---

## 🔑 Key Features

- **3 normalized PostgreSQL tables** — customers, services, billing
- **15 SQL exploratory analysis queries** — churn rate, segment analysis, window functions
- **Feature engineering in SQL** — tenure buckets, service count, revenue per month using CTEs and window functions
- **Full ML pipeline** — preprocessing, SMOTE for class imbalance, hyperparameter tuning with RandomizedSearchCV
- **Model explainability** — SHAP values to identify top churn drivers
- **Production ready REST API** — FastAPI endpoint serving real time predictions with risk tier classification

---

## 💡 Key Insights

- Month-to-month customers churn at **42%** vs only **3%** for two year contracts
- Fiber optic internet service has the **highest churn rate** across all segments
- Electronic check payment method correlates with **highest churn probability**
- New customers (0-12 months tenure) churn at **significantly higher rates** than loyal customers
- Churned customers pay **higher monthly charges** on average than retained customers

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up PostgreSQL
- Create a database called `churn_project`
- Run the table creation script:
```bash
psql -U postgres -d churn_project -f sql/create_tables.sql
```

### 4. Run the notebook
Open and run `notebooks/churn_analysis.ipynb` top to bottom. This will:
- Clean and load data into PostgreSQL
- Engineer features
- Train and evaluate all models
- Generate `model.pkl`, `scaler.pkl`, `ohe.pkl`, `le.pkl` in the models/ folder

### 5. Run the API
```bash
cd api
uvicorn app:app --reload
```

API will be live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## 📡 API Usage

**Endpoint:** `POST /predict`

**Sample Request:**
```json
{
  "tenure": 12,
  "monthlycharges": 65.5,
  "totalcharges": 786.0,
  "total_services": 3,
  "avg_revenue_per_month": 65.5,
  "contract": "Month-to-month",
  "paymentmethod": "Electronic check",
  "gender": "Male",
  "partner": "No",
  "dependents": "No",
  "seniorcitizen": 0,
  "tenure_segment": "New"
}
```

**Sample Response:**
```json
{
  "churn_probability": 0.7823,
  "churn_prediction": 1,
  "risk_tier": "High",
  "message": "This customer is likely to churn"
}
```

---

## 📈 SHAP Feature Importance

SHAP values reveal the most important features driving churn predictions:
- **Tenure** — longer tenure significantly reduces churn probability
- **Monthly Charges** — higher charges increase churn risk
- **Contract Type** — month-to-month contracts are the strongest churn predictor
- **Total Services** — customers with more services are less likely to churn

---

## 👤 Author

**Your Name**
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License.
