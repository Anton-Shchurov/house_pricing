# 🏠 House Prices Prediction | Advanced Regression Techniques

A machine learning project for the Kaggle competition "House Prices: Advanced Regression Techniques". The goal is to predict residential property prices based on various house characteristics.

## 📊 Project Overview

This project represents a complete data science and machine learning pipeline:
- **Exploratory Data Analysis (EDA)**
- **Data Preprocessing and Feature Engineering**
- **Comparison of Various Machine Learning Algorithms**
- **Hyperparameter Optimization**
- **Final Model Creation for Predictions**

## 🗂️ Project Structure

```
house_prices/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Processed datasets
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data Preprocessing
│   └── 03_modeling.ipynb      # Modeling and Evaluation
├── results/
│   ├── figures/               # Plots and visualizations
│   ├── submissions/           # Kaggle submission files
│   ├── best_pipeline.pkl      # Best trained model
│   └── preprocessor.pkl       # Data preprocessor
├── requirements.txt           # Project dependencies
└── README.md                 # Project description
```

## 🔍 Methodology

### 1. Exploratory Data Analysis (EDA)
- Feature distribution analysis
- Correlation analysis between variables
- Outlier and anomaly detection
- Visualization of relationships with target variable

### 2. Data Preprocessing
- **Removal of low-information features** (>98% identical values)
- **Missing value handling**:
  - "NA" for categorical features (meaning absence)
  - 0 for numeric features (meaning absence)
  - Neighborhood-wise median for `LotFrontage`
- **Binary transformations** of rare categorical features
- **Feature engineering**:
  - House age, remodel age, garage age
  - Total porch area and bathroom count
  - Total house area
- **Outlier removal** (SalePrice > $500,000)
- **Log transformation** of target variable and highly skewed features

### 3. Modeling
The following algorithms were tested:
- **ElasticNet** - linear regression with regularization
- **Random Forest** - ensemble of decision trees
- **XGBoost** - gradient boosting
- **Polynomial Ridge** - polynomial regression with regularization
- **MLP (Neural Network)** - multi-layer perceptron

### 4. Optimization
- **RandomizedSearchCV** for hyperparameter tuning
- **5-fold Cross Validation** for model evaluation
- Evaluation metric: **RMSE** (Root Mean Square Error)

## 📈 Results

Best performing models:
- **ElasticNet**: CV RMSE ≈ 0.1266
- **XGBoost**: High accuracy with optimized parameters
- **Random Forest**: Stable cross-validation results

## 🛠️ Technologies and Libraries

- **Python 3.x**
- **Pandas** - data manipulation
- **NumPy** - numerical computing
- **Scikit-learn** - machine learning
- **XGBoost** - gradient boosting
- **Matplotlib/Seaborn** - visualization
- **Jupyter Notebook** - interactive development

## 🚀 Getting Started

1. **Clone the repository**:
```bash
git clone https://github.com/Anton-Shchurov/house_pricing.git
cd house_pricing
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run notebooks in order**:
   - `01_eda.ipynb` - for exploratory data analysis
   - `02_preprocessing.ipynb` - for data preprocessing
   - `03_modeling.ipynb` - for model training

## 📋 Key Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 🎯 Key Project Features

- **Comprehensive data preprocessing** tailored for real estate data
- **Feature engineering** to create more informative variables
- **Multiple algorithm comparison** for optimal model selection
- **Systematic hyperparameter optimization**
- **Reproducible results** with fixed random_state
- **Modular structure** for easy understanding and modification

## 📊 Visualizations

The project includes numerous visualizations:
- Feature distributions before and after transformations
- Correlation matrices
- Scatter plots for relationship analysis
- Model performance comparisons

## 📞 Contact

If you have any questions or suggestions about the project, feel free to reach out!
LinkedIn: www.linkedin.com/in/anton-shchurov

---
*Project created as part of machine learning studies and Kaggle competition participation*
