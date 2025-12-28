📌 House Price Prediction using Machine Learning
Overview

This project builds an end-to-end machine learning pipeline to predict house sale prices using the Ames Housing dataset. The goal is not only to achieve reasonable predictive performance, but also to demonstrate a clean ML workflow including preprocessing, model training, and evaluation.

Problem Statement

Predict the final sale price of residential houses based on various numerical and categorical features such as location, size, quality, and condition.

Dataset

Ames Housing Dataset (Kaggle)

Contains both numerical and categorical features

Target variable: SalePrice

Approach

Performed basic exploratory data analysis to understand feature distributions and missing values

Used a preprocessing pipeline with:

Median imputation for numerical features

Most-frequent imputation + one-hot encoding for categorical features

Built a unified pipeline using ColumnTransformer and Pipeline

Trained a RandomForestRegressor model

Evaluated performance using Mean Absolute Error (MAE)

Model & Evaluation

Model: Random Forest Regressor

Metric: Mean Absolute Error (MAE)

Validation MAE: ~17,500

MAE was chosen because it is easy to interpret in real-world monetary terms and provides a clear measure of average prediction error.

Results

The model achieves a reasonable baseline performance without data leakage, demonstrating a correct and reproducible ML workflow. Further improvements can be made through hyperparameter tuning and advanced feature engineering.

Future Work

Hyperparameter tuning

Model explainability (feature importance, SHAP)

Deployment using Streamlit or cloud services

Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn