import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('C:/Users/ARKA ROY/OneDrive/Desktop/ml projects/loan approval prediction/lgbm_best_model.pkl')

# Title of the web app
st.title("Loan Approval Prediction")

# Create form inputs
st.header("Enter the details:")

    
interest_rate = st.slider('Interest Rate', min_value=0.0, max_value=20.0, step=0.01)
loan_amount = st.number_input('Loan Amount')
st.write("The loan amount is ",f"{loan_amount:,}")
income =  st.number_input('Annual Income')
st.write("Annual income is ",f"{income:,}")
months_employed = st.slider('Months Employed', min_value=0.0,max_value=1000.0,step=1.0)
credit_score = st.slider('Credit Score', min_value=300.0, max_value=850.0, step=1.0)
age = st.slider('Age', min_value=12.0, max_value=150.0, step=1.0)
dti_ratio = st.slider('Debt-to-Income Ratio', min_value=0.0, max_value=10.0, step=0.01)
# Create a button to submit the input data
if st.button('Predict'):
    # Create feature array for prediction
    features = np.array([[interest_rate, loan_amount, income, months_employed, credit_score, age, dti_ratio]])

    # Predict using the loaded model
    prediction = model.predict(features)

    # Assuming a binary classification where 1 means 'default' and 0 means 'no default'
    result = 'Default' if prediction[0] == 1 else 'No Default'

    # Display the result
    st.subheader("Prediction Result:")
    if(result=='Default'):
          st.markdown(f"<div style='color: red; font-size: 32px;'>{result+ '!!!'}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color: green; font-size: 32px;'>{result}div>", unsafe_allow_html=True)     
