# 🚕 NYC Taxi Demand Predictor

**A real-time machine learning system for predicting taxi demand in New York City**

[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-blue)](https://taxi-prediction.streamlit.app/)
[![Monitoring](https://img.shields.io/badge/Live-Monitoring-green)](https://frontedmonitoring-taxi.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)](https://streamlit.io/)

## 🎯 Problem Statement

Taxi demand prediction is crucial for optimizing fleet management, reducing wait times, and improving urban transportation efficiency. This project solves the challenge of predicting hourly taxi demand across different NYC zones using historical ride data and machine learning.

## 🚀 Solution Overview

This end-to-end ML system:
- **Predicts hourly taxi demand** for each NYC taxi zone
- **Processes 3+ years** of NYC taxi ride data (2022-2025)
- **Delivers real-time predictions** through interactive dashboards
- **Monitors model performance** with comprehensive metrics
- **Scales automatically** using feature stores and model registry

## 🏗️ System Architecture

```
NYC Taxi Data → Feature Engineering → ML Model → Predictions → Dashboard
      ↓                ↓                ↓           ↓           ↓
   Raw Rides    Time Series Data    LightGBM    Feature Store  Streamlit
```

### Key Components:
- **Data Pipeline**: Automated ETL processing of NYC taxi data
- **Feature Store**: Hopsworks-powered feature management
- **ML Pipeline**: LightGBM with hyperparameter optimization
- **Model Registry**: Versioned model deployment
- **Monitoring**: Real-time performance tracking
- **Frontend**: Interactive prediction dashboard

## 🔧 Tech Stack

- **ML Framework**: LightGBM, XGBoost, Scikit-learn
- **Feature Store**: Hopsworks
- **Frontend**: Streamlit, Plotly, PyDeck
- **Data Processing**: Pandas, NumPy, GeoPandas
- **Hyperparameter Tuning**: Optuna
- **Orchestration**: Poetry, Make
- **Monitoring**: Custom metrics tracking

## 📊 Live Applications

### 🎯 [Prediction Dashboard](https://taxi-prediction.streamlit.app/)
- Real-time taxi demand predictions
- Interactive NYC map visualization
- Top 10 locations with highest predicted demand
- Historical time-series analysis

### 📈 [Monitoring Dashboard](https://frontedmonitoring-taxi.streamlit.app/)
- Model performance metrics (MAE)
- Hour-by-hour accuracy analysis
- Location-specific performance tracking
- Prediction vs actual comparison

## 🚦 Getting Started

### Prerequisites
- Python 3.9+
- Poetry (for dependency management)
- Hopsworks account (for feature store)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/taxi-demand-predictor.git
   cd taxi-demand-predictor
   ```

2. **Install dependencies**
   ```bash
   make init
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   HOPSWORKS_PROJECT_NAME=your_project_name
   HOPSWORKS_API_KEY=your_api_key
   ```

### Usage

#### 🚨 First Time Setup (Required)

**Before running any other commands**, you must populate the feature store with historical data:

```bash
# ⚠️ REQUIRED: Backfill historical data (run once)
make backfill
```

> **Why is this needed?** The ML model requires 28 days (672 hours) of historical data for training and inference. This is standard practice in time-series ML systems.

#### 🔄 Regular Workflow

After the initial backfill, use these commands for regular operations:

```bash
# Generate features and store in feature store
make features

# Train the model
make training

# Generate predictions
make inference

# Launch prediction dashboard
make frontend-app

# Launch monitoring dashboard
make monitoring-app
```

#### 📋 Complete Setup Checklist

1. ✅ `make init` - Install dependencies
2. ✅ Configure `.env` file
3. ✅ `make backfill` - **Essential first step**
4. ✅ `make features` - Update with recent data
5. ✅ `make training` - Train the model
6. ✅ `make inference` - Generate predictions

## 🔍 Project Structure

```
├── data/                    # Raw and processed data
├── models/                  # Trained model artifacts
├── notebooks/               # Jupyter notebooks for analysis
├── scripts/                 # Pipeline scripts
├── src/                     # Source code
│   ├── config.py           # Configuration settings
│   ├── feature_store_api.py # Feature store interactions
│   ├── inference.py        # Prediction logic
│   ├── fronted.py          # Main dashboard
│   └── fronted_monitoring.py # Monitoring dashboard
├── tests/                   # Test files
├── Makefile                # Automation commands
└── pyproject.toml          # Dependencies
```

## 🧠 Model Performance

- **Algorithm**: LightGBM with feature engineering
- **Features**: 24×28 = 672 hourly historical features
- **Target**: Maximum MAE threshold of 30.0
- **Optimization**: Optuna-powered hyperparameter tuning
- **Validation**: Time-series cross-validation

## 🔮 Key Features

- **Real-time Predictions**: Hourly demand forecasting
- **Geospatial Visualization**: Interactive NYC taxi zone maps
- **Time Series Analysis**: Historical demand patterns
- **Model Monitoring**: Performance tracking and alerts
- **Automated Pipelines**: End-to-end ML workflow
- **Scalable Architecture**: Feature store and model registry

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML system design
- Feature engineering for time series
- Model deployment and monitoring
- Real-time dashboard development
- MLOps best practices
- Geospatial data visualization

## 🔧 Troubleshooting

### Common Issues

**❌ Empty training set error**
```bash
# Solution: Run backfill first
make backfill
```

**❌ Inference fails with data errors**
```bash
# Solution: Ensure backfill has been executed
make backfill
make features
```

**❌ Missing historical data**
- **Cause**: Backfill step was skipped
- **Solution**: Always run `make backfill` before other commands

### Important Notes

- **One-time operation**: `make backfill` only needs to be run once unless you want to refresh all historical data
- **Data dependencies**: All ML operations (training, inference) depend on the historical data populated by backfill

## 🙏 Acknowledgments

This project is based on the excellent tutorial by [Pau Labarta Bajo](https://www.linkedin.com/in/pau-labarta-bajo-4432074b). Special thanks for providing the foundational architecture and approach for this taxi demand prediction system.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

