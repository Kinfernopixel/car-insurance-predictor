import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# 1. Load data
df = pd.read_csv('data/raw/car_insurance_premium_dataset.csv')

# 2. Data cleanup
df['Car Manufacturing Year'] = np.where(df['Car Manufacturing Year'] > 2024, 2024, df['Car Manufacturing Year'])
df['Car Age'] = 2024 - df['Car Manufacturing Year']

# 3. Features and target
feature_cols = [
    'Driver Age', 'Driver Experience', 'Previous Accidents',
    'Annual Mileage (x1000 km)', 'Car Age'
]
X = df[feature_cols]
y = df['Insurance Premium ($)']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model training & evaluation
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lr = linreg.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def print_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}:")
    print(f"  Mean Absolute Error: {mae:.2f}")
    print(f"  R² Score: {r2:.4f}")
    print()

print_metrics(y_test, y_pred_lr, "Linear Regression")
print_metrics(y_test, y_pred_rf, "Random Forest Regressor")

# 6. Save the trained Linear Regression model and feature columns
os.makedirs('models', exist_ok=True)
joblib.dump(linreg, 'models/car_insurance_linreg.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')
