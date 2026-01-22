# Customer Churn Analytics - End-to-End MLOps Project

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.86-brightgreen.svg)](#)

## 🎯 Project Overview

 **Customer Churn Prediction & Retention Analytics** system for Telecom businesses. This end-to-end ML project demonstrates industry-standard practices from data generation to deployment, monitoring, and testing.



---

## ✨ Features

✅ **Synthetic Data Generation**: 5,000 customers with 12 months of realistic behavior  
✅ **Relational Database**: Multi-table SQLite with optimized indexes  
✅ **Feature Engineering**: SQL aggregations + interaction features + log transforms  
✅ **Model Comparison**: Logistic Regression, Random Forest, GBM, XGBoost  
✅ **Production Pipeline**: Scikit-learn pipeline preventing data leakage  
✅ **Explainability**: SHAP values + global feature importance  
✅ **Customer Segmentation**: K-Means clustering (k=4) on behavioral features  
✅ **Retention Strategy**: Rule-based recommendation engine  
✅ **REST API**: FastAPI with Pydantic validation  
✅ **Web Dashboard**: Multi-page Streamlit app with analytics  
✅ **BI Integration**: Tableau/Power BI ready dataset + KPI definitions  
✅ **MLOps**: Drift detection (PSI), baseline monitoring  
✅ **Testing**: Pytest suite (data quality, API, model inference)  
✅ **Documentation**: Comprehensive architecture, business, ML docs  

---

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
# Generate data & build ML dataset
python generate_dataset.py
python src/db_setup.py
python src/load_to_sqlite.py
python src/build_final_dataset.py

# Train model
python src/preprocess.py
python src/train_models.py
python src/explain.py
```

### 3. Launch API
```bash
pip install -r requirements_api.txt
uvicorn app.main:app --reload
```

### 4. Launch Dashboard
```bash
pip install -r requirements_streamlit.txt
streamlit run streamlit_app/Home.py
```

---

## 📁 Project Structure

```
churn_project/
├── data/
│   ├── raw/              # Generated CSV files (customers, transactions, events)
│   └── processed/        # Cleaned datasets
├── src/                  # Core Python modules
│   ├── config.py
│   ├── db_setup.py
│   ├── load_to_sqlite.py
│   ├── build_features_sql.py
│   ├── build_final_dataset.py
│   ├── eda_utils.py
│   ├── data_quality_checks.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── segmentation.py
│   ├── retention_strategy.py
│   ├── data_contract.py
│   └── build_bi_dataset.py
├── outputs/
│   ├── models/           # best_model.pkl
│   ├── plots/            # ROC, confusion matrix, feature importance
│   ├── eda_report.txt
│   ├── metrics.json
│   ├── customer_scores.csv
│   ├── segments.csv
│   └── retention_actions.csv
├── app/                  # FastAPI
│   ├── main.py
│   ├── schemas.py
│   └── utils.py
├── streamlit_app/        # Streamlit Dashboard
│   ├── Home.py
│   └── pages/
│       ├── 1_Single_Prediction.py
│       ├── 2_Batch_Prediction.py
│       └── 3_Analytics_Dashboard.py
├── bi/                   # Business Intelligence
│   ├── tableau_ready_dataset.csv
│   ├── churn_kpis.md
│   └── dashboard_wireframe.md
├── monitoring/           # MLOps Monitoring
│   ├── monitor.py
│   ├── drift.py
│   ├── baseline_stats.json
│   └── drift_report.txt
├── tests/                # Pytest Suite
│   ├── test_data_quality.py
│   ├── test_api.py
│   └── test_model_inference.py
├── docs/                 # Documentation
│   ├── architecture.md
│   ├── business_summary.md
│   ├── ml_summary.md
│   ├── testing_guide.md
│   └── results.md
├── churn.db              # SQLite database
├── requirements.txt      # Core dependencies
├── requirements_api.txt
├── requirements_streamlit.txt
├── requirements_test.txt
└── README.md
```

---

## 📊 Dataset

### Raw Tables (SQLite)
- **customers**: Demographics (5,000 rows)
- **subscriptions**: Subscription lifecycle
- **transactions**: Payment history (60K+ rows)
- **app_events_daily**: Daily app usage (1.8M+ rows)
- **support_tickets**: Customer support interactions
- **churn_labels**: Ground truth churn flags

### ML Dataset (17 features)
```
customer_id, plan_type, monthly_price, age, gender, location, 
device_type, acquisition_channel, auto_renew, 
total_sessions_30d, avg_session_minutes_30d, total_crashes_30d, 
failed_payments_30d, total_amount_success_30d, 
support_tickets_30d, avg_resolution_time_30d, 
churn
```

**Churn Rate**: ~22% (realistic for SaaS/OTT)

---

## 🤖 ML Pipeline

### Training Flow
1. `generate_dataset.py` → data/raw/*.csv
2. `src/db_setup.py` → churn.db
3. `src/load_to_sqlite.py` → CSV → DB
4. `src/build_final_dataset.py` → outputs/final_churn_dataset.csv
5. `src/preprocess.py` → outputs/cleaned_dataset.csv
6. `src/train_models.py` → outputs/models/best_model.pkl
7. `src/explain.py` → outputs/customer_scores.csv
8. `src/segmentation.py` → outputs/segments.csv
9. `src/retention_strategy.py` → outputs/retention_actions.csv

### Models Compared
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- **XGBoost** (best: 81% acc, 0.86 ROC-AUC)

---

## 🔌 API Usage

### Start Server
```bash
uvicorn app.main:app --reload --port 8000
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C001",
    "plan_type": "Premium",
    "monthly_price": 999,
    "age": 28,
    "gender": "Male",
    "location": "Mumbai",
    "device_type": "Android",
    "acquisition_channel": "Organic",
    "auto_renew": 1,
    "total_sessions_30d": 45,
    "avg_session_minutes_30d": 25.5,
    "total_crashes_30d": 2,
    "failed_payments_30d": 0,
    "total_amount_success_30d": 999.0,
    "support_tickets_30d": 0,
    "avg_resolution_time_30d": 0.0
  }'
```

**Response**:
```json
{
  "customer_id": "C001",
  "churn_probability": 0.15,
  "churn_prediction": 0,
  "risk_category": "Low"
}
```

See [`README_API.md`](README_API.md) for full documentation.

---

## 📱 Streamlit Dashboard

```bash
streamlit run streamlit_app/Home.py
```

**Pages**:
1. **Home**: Project overview & features
2. **Single Prediction**: Form-based prediction with live API calls
3. **Batch Prediction**: CSV upload → predictions → download results
4. **Analytics Dashboard**: Interactive churn insights (Plotly charts)

See [`README_STREAMLIT.md`](README_STREAMLIT.md) for details.

---

## 📈 BI Integration

### Generate BI Dataset
```bash
python src/build_bi_dataset.py
```

**Output**: `bi/tableau_ready_dataset.csv`

**Docs**: 
- [`bi/churn_kpis.md`](bi/churn_kpis.md) - KPI definitions
- [`bi/dashboard_wireframe.md`](bi/dashboard_wireframe.md) - Dashboard design

**Import into**: Tableau, Power BI, Looker

---

## 🔍 Monitoring & Drift Detection

### Generate Baseline
```bash
python monitoring/monitor.py
```

### Detect Drift (PSI)
```bash
python monitoring/drift.py
```

**Interpretation**:
- PSI < 0.1: No drift
- 0.1 ≤ PSI < 0.2: Moderate drift
- PSI ≥ 0.2: **Retraining recommended**

---

## ✅ Testing

### Install Test Dependencies
```bash
pip install -r requirements_test.txt
```

### Run All Tests
```bash
pytest tests/ -v
```

### Test Categories
- **Data Quality** (`test_data_quality.py`): Schema, constraints, business rules
- **API** (`test_api.py`): Endpoint validation (requires API running)
- **Model** (`test_model_inference.py`): Inference correctness, performance thresholds

### Data Contract Validation
```bash
python src/data_contract.py
```

See [`docs/testing_guide.md`](docs/testing_guide.md) for CI/CD integration.

---

## 📚 Documentation

Comprehensive docs in `docs/`:

- **[architecture.md](docs/architecture.md)**: System design, data flow, deployment
- **[business_summary.md](docs/business_summary.md)**: ROI analysis, use cases, KPIs
- **[ml_summary.md](docs/ml_summary.md)**: Model methodology, evaluation, tuning
- **[testing_guide.md](docs/testing_guide.md)**: Pytest usage, test coverage
- **[results.md](docs/results.md)**: Performance metrics, key findings

---

## 🎯 Key Results

### Model Performance (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | **81%** |
| ROC-AUC | **0.86** |
| Precision | **76%** |
| Recall | **71%** |
| F1-Score | **73%** |

### Top Churn Drivers
1. **failed_payments_30d** (25% importance)
2. **total_sessions_30d** (20%)
3. **auto_renew** (15%)
4. **avg_session_minutes_30d** (12%)
5. **support_tickets_30d** (10%)

### Business Impact
- **Revenue at Risk**: ₹4.5M (high-risk customers)
- **Projected Annual Savings**: ₹1.82M (with 30% churn reduction)
- **ROI**: 9.5x
- **Precision**: 76% (minimizes wasted retention spend)

See [`docs/results.md`](docs/results.md) for full analysis.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Data | pandas, numpy, SQLite |
| ML | scikit-learn, XGBoost |
| API | FastAPI, Uvicorn, Pydantic |
| Web App | Streamlit |
| Visualization | matplotlib, plotly |
| Testing | pytest |
| BI | Tableau / Power BI |

---

## 📖 10-Day Project Plan

### Day 1: Data Generation & DB Setup ✅
- Generated 5,000 customers with realistic behavior
- Created SQLite database with 6 tables
- Built ML-ready dataset with 30-day aggregations

### Day 2: EDA & Data Cleaning ✅
- Comprehensive EDA report
- Data quality checks
- Outlier capping and missing value imputation

### Day 3: Feature Engineering & Model Training ✅
- Interaction features and log transforms
- Trained 4 models with cross-validation
- F1-based threshold tuning

### Day 4: Explainability & Segmentation ✅
- SHAP values for local explanations
- K-Means customer segmentation
- Rule-based retention strategies

### Day 5: FastAPI Deployment ✅
- REST API with `/predict` and `/batch_predict`
- Pydantic validation
- API documentation

### Day 6: Streamlit Web App ✅
- Multi-page dashboard
- Single and batch predictions
- Interactive analytics

### Day 7: BI Integration ✅
- BI-ready dataset for Tableau/Power BI
- KPI definitions
- Dashboard wireframe

### Day 8: Monitoring & Drift ✅
- Baseline statistics generation
- PSI-based drift detection
- Prediction logging

### Day 9: Testing & QA ✅
- Pytest test suite (15+ tests)
- Data contract validation
- Automated testing guide

### Day 10: Documentation ✅
- Architecture documentation
- Business summary with ROI
- ML technical summary
- Results and key findings

---

## 💼 Business Value

### Problem
- 22% baseline churn rate
- 1,100 customers lost monthly
- ₹659K monthly revenue loss

### Solution
- Predictive model with 81% accuracy
- Target high-risk customers (850/month)
- 30% churn reduction through retention campaigns

### Impact
- **Annual Revenue Saved**: ₹1.82M
- **Campaign Cost**: ₹1.02M/year  
- **Net Savings**: ₹804K/year
- **ROI**: 9.5x

---

## 🚀 Future Enhancements

1. **Real-time Predictions**: Streaming data with Kafka
2. **A/B Testing**: Test retention strategies
3. **Automated Retraining**: Trigger retraining on drift
4. **Advanced Explainability**: LIME, Anchor
5. **Multi-Model Ensemble**: Combine multiple models
6. **Customer Lifetime Value**: Predict CLV alongside churn

---

## 📄 License

This project is created for educational and portfolio purposes.

---

## 👤 Author

**Data Scientist + ML Engineer**  
Project: Customer Churn & Retention Analytics 

---

## 🙏 Acknowledgments

Built with industry best practices in ML engineering, MLOps, and software development.

**Technologies**: Python, scikit-learn, XGBoost, FastAPI, Streamlit, SQLite, Tableau
