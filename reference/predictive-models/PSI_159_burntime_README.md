## Burn Time Estimation (Regression) Model (PSI-159)

This documentation outlines how to integrate and utilize the machine learning regression model built to predict the **burn time (sec)** (flame stability duration) of gas mixtures under specified experimental configurations.

---

## 1. What It Is

This model is a **Random Forest Regressor Pipeline** that takes experimental variables and predicts how long (in seconds) the mixture will burn.

### Model Performance (Test Set Evaluation)
* **Mean Absolute Error (MAE):** ~23.08 seconds
* **Root Mean Squared Error (RMSE):** ~36.14 seconds
* **R-squared ($R^2$) Score:** ~15.75%

### Key Predictor Insights
Based on feature importance metrics, the model relies on the following predictors in order of significance:
1. **Adiabatic Flame Temperature (`Tad (K)`)** — The strongest predictor of stability.
2. **Hydrocarbon Burn Rate (`HC burnrate (mg/s)`)**
3. **Fuel Selection (`Fuel_30% ethane`)**
4. **System Pressure (`approx p (bar)`)**
5. **Oxygen Flow Rate (`O2 flow rate (mg/s)`)**

---

## 2. File Assets

* `PSI_159_burntime_regressor_Predictive_Model.joblib`: The serialized scikit-learn pipeline containing the preprocessing transformers (imputation, encoding) and the trained Random Forest Regressor.

---

## 3. How to Integrate and Use

### Prerequisites
Make sure the standard environment libraries are installed:
```bash
pip install numpy pandas scikit-learn joblib
```

### Python Execution Template
Use this block to load the pipeline and perform a prediction on new conditions:

```python
import joblib
import pandas as pd
import numpy as np

# 1. Load the pre-trained pipeline
reg_model_path = 'PSI_159_burntime_regressor_Predictive_Model.joblib'
loaded_reg_model = joblib.load(reg_model_path)

# 2. Setup inputs matching the pipeline features
features = [
    'Flame Type', 'Fuel', 'approx p (bar)', 'Tad (K)',
    'HC burnrate (mg/s)', 'O2 flow rate (mg/s)', 'XO2', 'Xf'
]

# 3. Create raw input dataframe
new_mixture = pd.DataFrame([
    {
        'Flame Type': 'Normal',
        'Fuel': '30% ethane',
        'approx p (bar)': 0.506650,
        'Tad (K)': 2483.47,
        'HC burnrate (mg/s)': 0.75,
        'O2 flow rate (mg/s)': 0.85,
        'XO2': 0.15,
        'Xf': 0.300
    }
])

# Standardize string error/Excel division formats to floats
numeric_cols = ['approx p (bar)', 'Tad (K)', 'HC burnrate (mg/s)', 'O2 flow rate (mg/s)', 'XO2', 'Xf']
for col in numeric_cols:
    new_mixture[col] = pd.to_numeric(new_mixture[col], errors='coerce')

# 4. Predict Burn Time
predicted_time = loaded_reg_model.predict(new_mixture[features])[0]
print(f"Estimated Burn Time: {predicted_time:.2f} seconds")
```