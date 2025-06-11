import joblib
import pandas as pd

# Load the saved model and feature columns
linreg = joblib.load('models/car_insurance_linreg.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

def predict_premium(input_dict):
    # input_dict should be a dictionary of feature_name: value
    X_new = pd.DataFrame([input_dict])[feature_cols]
    return linreg.predict(X_new)[0]

# Example use:
example_input = {
    'Driver Age': 28,
    'Driver Experience': 4,
    'Previous Accidents': 2,
    'Annual Mileage (x1000 km)': 18,
    'Car Age': 3
}
print(f"Predicted Premium: ${predict_premium(example_input):.2f}")
