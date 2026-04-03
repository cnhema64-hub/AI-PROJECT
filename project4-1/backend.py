
# 1. Import Libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib


# 2. Load Model

model = joblib.load("rf_model.pkl")
model1 = joblib.load("lr_model1.pkl")
scaler = joblib.load('scaler.pkl')


# 3. App

app = Flask(__name__)


# Updated feature columns
@app.route('/')
def home():
    return render_template('index.html')
# Home Page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
    
        amount = float(request.form['amount'])
        age = int(request.form['age'])
        velocity = float(request.form['velocity'])
        time = int(request.form['time'])
        merchant_category = float(request.form['merchant_category'])
        location_mismatch = int(request.form['location_mismatch'])
        device_trust_score = float(request.form['device_trust_score'])
        foreign_transaction = int(request.form['foreign_transaction'])
        
        

     

        # Create input list (IMPORTANT: match model features)
        features = [[amount, time, foreign_transaction, location_mismatch, device_trust_score, velocity, age, merchant_category]]
        


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