# 📘 Machine Learning Regression — Implemented From Scratch

This repository contains a complete implementation of **Linear Regression** and **Polynomial Regression** *from scratch* using only NumPy.  
It also includes two full applied machine learning pipelines:

- **House Price Prediction**
- **City Temperature Modeling and Seasonality Analysis**

The project demonstrates theoretical understanding, model implementation, data preprocessing, visualization, and full evaluation pipelines.

## 📁 Project Structure

```
ml-regression-from-scratch/
│
├── src/
│   ├── linear_regression.py
│   ├── polynomial_fitting.py
│   ├── house_price_prediction.py
│   ├── city_temperature_prediction.py
│   └── __init__.py
│
├── docs/
│   ├── Answers.pdf
│   └── graphs/
│
├── data/
│   ├── house_prices.csv
│   ├── city_temperature.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Features

### 1. Linear Regression (From Scratch)
Implements the closed-form OLS solution:

\[
\hat{\beta} = (X^TX)^{-1} X^Ty
\]

Includes intercept handling, prediction, and MSE loss.

### 2. Polynomial Regression
Extends linear regression using a Vandermonde matrix for polynomial fitting.

### 3. House Price Prediction Pipeline
Includes:
- Preprocessing  
- Feature cleaning  
- Pearson correlation analysis  
- Mean loss vs. train size with confidence intervals  

### 4. City Temperature Modeling Pipeline
Includes:
- Day-of-year transformation  
- Seasonal visualization  
- Polynomial fitting for k=1…10  
- Cross‑country generalization evaluation  

## 📊 Visual Outputs
The project generates:
- Scatter plots  
- STD per month  
- Polynomial error comparison  
- Feature correlations  
- Training-size analysis  

## 🧠 Theoretical Component
Located in `docs/Answers.pdf`, covering:
- Linear algebra  
- Least squares derivations  
- Polynomial approximation theory  
- Multivariate calculus  
- SVD connections  

## 📦 Installation
```
pip install -r requirements.txt
```

## ▶️ Usage
```
python src/city_temperature_prediction.py
python src/house_price_prediction.py
```

## 🛠 Technologies
Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn (split only).

## 🎯 Learning Outcomes
- Implementing ML algorithms manually  
- Strong math foundations  
- Data processing + visualization  
- Model evaluation  
- Clean reproducible ML pipelines  

## 📘 License
MIT License.

## 🙌 Author
**Yair Mahfud**
