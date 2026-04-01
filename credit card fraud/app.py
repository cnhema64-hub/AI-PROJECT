from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models and scaler
model = joblib.load("random_forest_fraud_model.pkl")
model1 = joblib.load("logistic_fraud_model1.pkl")
scaler = joblib.load('scaler.pkl')

# Updated feature columns
feature_columns = [
    'distance_from_home',
    'distance_from_last_transaction',
    'ratio_to_median_purchase_price',
    'repeat_retailer',
    'used_chip',
    'used_pin_number',
    'online_order'
]
@app.route('/')
def home():
    return render_template('index.html')
# Home Page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        V1 = float(request.form['V1'])
        V2 = float(request.form['V2'])
        V3 = float(request.form['V3'])
        V4 = float(request.form['V4'])
        V5 = float(request.form['V5'])
        V6= float(request.form['V6'])
        V7= float(request.form['V7'])

        #Amount = float(request.form['Amount'])

        # Create input list (IMPORTANT: match model features)
        features = [[V1, V2, V3,V4,V5,V6,V7]]

        # Convert to numpy
        import numpy as np
        features = np.array(features)

        # Scale if needed
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        result = "Fraudulent Transaction" if prediction == 1 else "good Transaction"

        return render_template(
            'index.html',
            model_prediction=result,
            model_probability=round(prob * 100, 2)
        )

    except Exception as e:
        return render_template('index.html', error=str(e))
    
if __name__ == "__main__":
    app.run(debug=True)