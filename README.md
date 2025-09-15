# Predicting Market Value of Footballers 🏆

A comprehensive machine learning project that predicts the market value of football players using various performance metrics, player attributes, and statistical analysis. The project implements multiple regression algorithms with hyperparameter tuning to achieve high accuracy in player valuation predictions.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project leverages machine learning to predict football player market values, which is crucial for:
- **Transfer Market Analysis** - Evaluate fair transfer fees and identify market opportunities
- **Contract Negotiations** - Determine appropriate compensation structures
- **Investment Decisions** - Spot undervalued or overvalued players
- **Strategic Planning** - Support club management and scouting decisions

## 📊 Key Results

Our best performing model achieves:
- **R² Score**: 0.988 (98.8% variance explained)
- **Mean Absolute Error**: €232K
- **Root Mean Square Error**: €463K
- **Mean Absolute Percentage Error**: 6.81%

## ✨ Features

- **Multiple ML Algorithms** - Linear Regression, Random Forest, Gradient Boosting, Polynomial Regression, Lasso, Ridge
- **Hyperparameter Tuning** - Automated parameter optimization using RandomizedSearchCV
- **Feature Engineering** - Advanced feature selection and importance analysis
- **Ensemble Methods** - Voting and Stacking regressors for improved performance
- **Model Persistence** - Save and load trained models
- **Comprehensive Evaluation** - Multiple metrics for robust assessment

## 📁 Project Structure

```
Predicting-Market-Value-Footballers/
├── Web Scaping/
│   └── web_scraping.ipynb          # Data collection notebook
├── data/
│   ├── out.csv                     # Processed dataset
│   ├── players_all.csv             # Complete player data
│   └── test.csv                    # Test dataset
├── data_preparation/
│   └── data_prepare.ipynb          # Data preprocessing
├── main.ipynb                      # Main analysis and modeling
├── gradient_boosting_with_most_im... # Best model file
└── README.md
```

## 🛠️ Methodology

### 1. Data Collection & Preprocessing
- **Web scraping from SoFiFA.com** - Automated collection of 60+ player attributes across multiple pages
- **Comprehensive dataset** - Player statistics including ratings, physical attributes, skills, and market values
- **Data cleaning and feature engineering** - Processing scraped data into ML-ready format
- **Handling missing values and outliers** - Data quality assurance and preprocessing
- **Feature scaling and normalization** - Preparation for machine learning algorithms

### 2. Model Development
The project implements and compares multiple algorithms:

#### Linear Models
- **Linear Regression** - R²: 0.891, MAE: €848K
- **Lasso Regression** - R²: 0.868, MAE: €916K  
- **Ridge Regression** - R²: 0.868, MAE: €916K
- **Polynomial Regression** - R²: 0.918, MAE: €758K

#### Ensemble Methods
- **Random Forest** - R²: 0.976, MAE: €302K
- **Gradient Boosting** - R²: 0.979, MAE: €339K
- **Tuned Random Forest** - R²: 0.961, MAE: €419K
- **Tuned Gradient Boosting** - R²: 0.983, MAE: €260K

#### Advanced Ensembles
- **Voting Regressor** - R²: 0.978, MAE: €306K
- **Stacking Regressor** - R²: 0.988, MAE: €227K

### 3. Feature Importance Analysis

The most important features for predicting market value are:
- **Age** - Player's current age
- **Overall Rating** - FIFA overall skill rating
- **Potential** - Maximum potential rating
- **Best Overall** - Peak overall rating achieved
- **Growth** - Difference between potential and current rating
- **Dribbling/Reflexes** - Technical skills
- **Wages** - Current salary information
- **Release Clause** - Contract release clause value

### 4. Model Optimization

#### Hyperparameter Tuning
- **RandomizedSearchCV** with 5-fold cross-validation
- **50 parameter combinations** tested for each model
- **Optimized parameters** for Random Forest and Gradient Boosting

#### Best Model Configuration
**Gradient Boosting with Selected Features:**
```python
GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42
)
```

## 📈 Usage

### Running the Complete Analysis

1. **Open the main notebook**
```bash
jupyter notebook main.ipynb
```

2. **Execute cells sequentially** to:
   - Load and explore the dataset
   - Train multiple models
   - Compare performance metrics
   - Analyze feature importance
   - Generate predictions

### Using the Trained Model

```python
from joblib import load
import pandas as pd

# Load the best model
model = load('gradient_boosting_with_most_important_features_best_model.joblib')

# Prepare your data (ensure same features as training)
# new_player_data = pd.DataFrame({...})

# Make predictions
predicted_value = model.predict(new_player_data)
print(f"Predicted market value: €{predicted_value[0]:,.0f}")
```

### Key Features Required for Prediction

```python
required_features = [
    'Age',
    'Overall rating', 
    'Potential',
    'Best overall',
    'Growth',
    'Dribbling / Reflexes',
    'new_wages',
    'new_release_clause'
]
```

## 📊 Model Performance Comparison

| Model | R² Score | MAE (€) | RMSE (€) | MAPE (%) |
|-------|----------|---------|----------|----------|
| Linear Regression | 0.891 | 848K | 1,348K | 72.0 |
| Random Forest | 0.976 | 302K | 641K | 7.6 |
| Gradient Boosting | 0.979 | 339K | 602K | 13.3 |
| **Tuned Gradient Boosting** | **0.983** | **260K** | **539K** | **7.4** |
| **GB with Selected Features** | **0.988** | **232K** | **463K** | **6.8** |
| Stacking Regressor | 0.988 | 227K | 456K | 6.7 |

## 🔍 Data Pipeline

### 1. Web Scraping (`Web Scaping/web_scraping.ipynb`)
- Automated data collection from football databases
- Player statistics, ratings, and market values
- Real-time data updates

### 2. Data Preparation (`data_preparation/data_prepare.ipynb`)
- Data cleaning and validation
- Feature engineering and selection
- Train/test split preparation

### 3. Model Training (`main.ipynb`)
- Comprehensive model comparison
- Hyperparameter optimization
- Performance evaluation and validation


## 🎯 Future Improvements

- **Deep Learning Models** - Neural networks for non-linear patterns
- **Time Series Analysis** - Player value trends over time
- **Real-time Predictions** - API integration for live data
- **Additional Features** - Injury history, team performance metrics
- **Multi-class Prediction** - Position-specific value models




## 📞 Contact

For questions or collaboration opportunities, please open an issue in the repository.

---
