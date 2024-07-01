from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('lgbm_best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    interest_rate = float(request.form['InterestRate'])
    loan_amount = float(request.form['LoanAmount'])
    income = float(request.form['Income'])
    months_employed = float(request.form['MonthsEmployed'])
    credit_score = float(request.form['CreditScore'])
    age = float(request.form['Age'])
    dti_ratio = float(request.form['DTIRatio'])

    # Create feature array for prediction
    features = np.array([[interest_rate, loan_amount, income, months_employed, credit_score, age, dti_ratio]])

    # Predict using the loaded model
    prediction = model.predict(features)

    # Assuming a binary classification where 1 means 'default' and 0 means 'no default'
    result = 'Default' if prediction[0] == 1 else 'No Default'

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
