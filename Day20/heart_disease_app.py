import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'heart_model_final.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

try:
    package = load_model()
    model = package['model']
    scaler = package['scaler']
    feature_names = package['feature_names']
    model_accuracy = package['accuracy']
    metrics = package.get('metrics', {})
except FileNotFoundError:
    st.error("Model file not found!")
    st.stop()

# Title
st.title("Heart Disease Prediction System")
st.markdown(f"**Model:** {package['model_name']} | **Accuracy:** {model_accuracy*100:.1f}%")
st.markdown("---")

# Sidebar — Patient Input
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 20, 80, 50)

sex = st.sidebar.selectbox("Sex", 
                           options=[0, 1],
                            format_func=lambda x: "Female" if x == 0 else "Male")

cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
    format_func=lambda x: ["Typical Angina",
                            "Atypical Angina",
                              "Non-anginal Pain",
                                "Asymptomatic"][x])

trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)


chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)


fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")

restecg = st.sidebar.selectbox("Resting ECG",
        options=[0, 1, 2],
    format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])

thalach = st.sidebar.slider("Max Heart Rate", 60, 210, 150)

exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes")

oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)

slope = st.sidebar.selectbox("ST Slope", options=[0, 1, 2],
    format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])


ca = st.sidebar.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3])

thal = st.sidebar.selectbox("Thalassemia", options=[0, 1, 2],
    format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

submit = st.sidebar.button("Predict", use_container_width=True)

# Predict
if submit:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                             thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Display Result
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Summary")
        patient_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_data[0]
        })
        st.dataframe(patient_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Prediction Result")
        st.markdown("")
        if prediction == 1:
            st.error("### HEART DISEASE RISK DETECTED")
            st.markdown("The model predicts this patient **has a risk of heart disease**.")
            st.markdown("**Recommendation:** Further medical examination is advised.")
        else:
            st.success("### HEALTHY — Low Risk")
            st.markdown("The model predicts this patient is **likely healthy**.")
            st.markdown("**Recommendation:** Continue regular check-ups.")

    # Model Performance Metrics
    st.markdown("---")
    st.subheader("Model Performance Metrics")
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
        m3.metric("Recall", f"{metrics['recall']*100:.1f}%")
        m4.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")
    else:
        st.info("Metrics not available. Re-train the model to see performance metrics.")
