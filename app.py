import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('best_knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title and instructions
st.set_page_config(page_title="Diabetes Risk Predictor")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter your health parameters below to check your risk of diabetes:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# Predict
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
