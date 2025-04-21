import streamlit as st
import numpy as np
import joblib
st.set_page_config(page_title="Health Risk Predictor", layout="wide", initial_sidebar_state="expanded")

# Sidebar to choose prediction model
st.sidebar.title("Choose Prediction Model")
selection = st.sidebar.radio("Select Model", ["Diabetes", "Heart Disease"])

# Load both models and scalers
diabetes_model = joblib.load('best_knn_model.pkl')
diabetes_scaler = joblib.load('scaler.pkl')

heart_model = joblib.load('heart_model.pkl')         # Make sure this file is in the repo
heart_scaler = joblib.load('heart_scaler.pkl')       # Make sure this file is in the repo

st.title(f"{selection} Risk Predictor")

# ===========================
# Diabetes Predictor Section
# ===========================
if selection == "Diabetes":
    st.markdown("Enter your health parameters to check your risk of diabetes:")

    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0, step=1)

    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        scaled_input = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

# ===========================
# Heart Disease Predictor Section
# ===========================
else:
    st.markdown("Enter your heart health parameters to check your risk of heart disease:")

    age = st.number_input("Age", min_value=0, step=1)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol Level")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved")
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)")
    slope = st.selectbox("Slope of the peak exercise ST segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)", [0, 1, 2])

    if st.button("Predict"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        scaled_input = heart_scaler.transform(input_data)
        prediction = heart_model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")
