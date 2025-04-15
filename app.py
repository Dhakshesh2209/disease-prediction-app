import streamlit as st
import pandas as pd
import joblib

# Load model and symptoms
model = joblib.load("disease_model.pkl")
symptom_list = pd.read_csv("symptom_list.csv").columns.tolist()

st.set_page_config(page_title="Disease Predictor", page_icon="🧠")

st.title("🧠 Disease Prediction from Symptoms")
st.markdown("Select the symptoms you’re experiencing and get a prediction.")

selected_symptoms = st.multiselect("Select your symptoms", symptom_list)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
        prediction = model.predict([input_vector])[0]
        st.success(f"🩺 Predicted Disease: **{prediction}**")
