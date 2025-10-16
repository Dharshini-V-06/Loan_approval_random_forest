import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------------------------
# Load model, encoder, and category mappings
# ------------------------------------------------
with open('loan_approval.pkl', 'rb') as f:
    le, model = pickle.load(f)

# If you also saved category mappings, uncomment this:
# le, model, categories = pickle.load(f)

# ------------------------------------------------
# Streamlit UI Setup
# ------------------------------------------------
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details below to predict loan approval status.")

# ------------------------------------------------
# Input Fields
# ------------------------------------------------
person_age = st.number_input("Person Age", min_value=18, max_value=80, value=25)
person_gender = st.selectbox("Gender", ['male', 'female'])
person_education = st.selectbox("Education", ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
person_income = st.number_input("Annual Income ($)", min_value=8000, max_value=10000000, value=60000)
person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=60, value=3)
person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=35000, value=5000)
loan_intent = st.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=5.0, max_value=25.0, value=12.0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=1, max_value=50, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ['No', 'Yes'])

# ------------------------------------------------
# Prepare input data for model
# ------------------------------------------------
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_gender': [person_gender],
    'person_education': [person_education],
    'person_income': [person_income],
    'person_emp_exp': [person_emp_exp],
    'person_home_ownership': [person_home_ownership],
    'loan_amnt': [loan_amnt],
    'loan_intent': [loan_intent],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
    'credit_score': [credit_score],
    'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
})

# ------------------------------------------------
# Encode categorical variables
# ------------------------------------------------
cat_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
for col in cat_cols:
    input_data[col] = le.fit_transform(input_data[col].astype(str))

# ------------------------------------------------
# Make prediction
# ------------------------------------------------
if st.button("🔍 Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Loan Approved with {probability*100:.2f}% confidence.")
    else:
        st.error(f"❌ Loan Rejected with {(1 - probability)*100:.2f}% confidence.")

    st.info("Model used: Random Forest Classifier")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
