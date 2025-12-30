# House Price Prediction using Machine Learning

## Overview
This project implements an end-to-end machine learning pipeline to predict residential house prices using the Ames Housing dataset. The focus of this project is not only on predictive performance, but also on building a clean, reproducible ML workflow and analyzing model behavior through proper evaluation and overfitting analysis.

## Problem Statement
The objective is to predict the final sale price of a house based on a combination of numerical and categorical features such as size, location, quality, and condition. This mirrors a common real-world regression problem in data science and machine learning.

## Dataset
- **Dataset:** Ames Housing Dataset (Kaggle)
- **Target Variable:** `SalePrice`
- **Features:** Mixture of numerical and categorical variables related to property attributes
- The dataset contains missing values and heterogeneous feature types, making it suitable for demonstrating preprocessing pipelines.

## Approach

### Data Preprocessing
- Split features into numerical and categorical columns
- Numerical features:
  - Imputed using median values to reduce sensitivity to outliers
- Categorical features:
  - Imputed using most frequent values
  - One-hot encoded to convert categories into numerical form
- All preprocessing steps are handled using a unified `ColumnTransformer` and `Pipeline` to avoid data leakage

### Models
- **Baseline Model:** Linear Regression  
- **Primary Model:** Random Forest Regressor

### Evaluation
- Metric used: **Mean Absolute Error (MAE)**
- MAE was chosen because it is easily interpretable in real-world monetary terms and provides a clear measure of average prediction error.

## Results

| Model | MAE |
|------|-----|
| Linear Regression | Higher than Random Forest |
| Random Forest (baseline) | ~17,500 |
| Random Forest (tuned) | ~17,500 |

Although hyperparameter tuning significantly reduced training error, validation performance remained similar to the baseline model, indicating overfitting and diminishing returns from increased model complexity.

## Overfitting Analysis
The tuned Random Forest model achieves a substantially lower training MAE (~7,200) compared to the validation MAE (~17,500). This gap indicates that the model fits the training data very well but does not generalize proportionally to unseen data. This behavior highlights the bias–variance tradeoff and the presence of inherent noise in the dataset.

## Project Structure
<img width="293" height="252" alt="image" src="https://github.com/user-attachments/assets/31e538e2-26be-47d8-89c4-082052ce2855" />


## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the notebook
   ```bash
   notebooks/01_house_price_prediction.ipynb
   ```
---
## Future Improvements 
1. Feature importance and permutation based explainability
2. Advanced feature engineering
3. Deployment using Streamlit or cloud services
4. Experimentation with gradient boosting models
---

## 👤 Author
Paranjay Das, 
BTech CSE (AI/ML), 
Aspiring Machine Learning Engineer
