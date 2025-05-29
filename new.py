import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
from gtts import gTTS
import base64
import g4f

# Load crop recommendation models only
@st.cache_resource
def load_models():
    crop_model = joblib.load('crop_recommender_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return crop_model, label_encoders

crop_model, label_encoders = load_models()

# App title
st.set_page_config(page_title="AgriSmart Assistant", layout="wide")
st.title('🌱 AgriSmart Assistant')
st.markdown("""
This intelligent assistant helps farmers with:
- Crop recommendations based on soil and weather conditions
- Agricultural advice through chatbot
""")

# Navigation
page = st.sidebar.radio("Navigation", ["Crop Recommendation", "Agri Chatbot"])

if page == "Crop Recommendation":
    st.header('Crop Recommendation System')
    
    with st.form("crop_recommendation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            district = st.selectbox('District', options=label_encoders['district'].classes_)
            soil_type = st.selectbox('Soil Type', options=label_encoders['soil_type'].classes_)
            ph_value = st.slider('pH Value', 0.0, 14.0, 6.5, 0.1)
            nitrogen = st.number_input('Nitrogen Level (kg/ha)', 0, 1000, 250)
            
        with col2:
            phosphorous = st.number_input('Phosphorous Level (kg/ha)', 0, 1000, 35)
            potassium = st.number_input('Potassium Level (kg/ha)', 0, 1000, 300)
            rainfall = st.number_input('Rainfall (mm)', 0, 5000, 650)
            temperature = st.number_input('Temperature (°C)', 0, 50, 28)
        
        farm_size = st.slider('Farm Size (hectares)', 0.1, 50.0, 3.5, 0.1)
        submit_button = st.form_submit_button("Get Crop Recommendation")

    if submit_button:
        district_enc = label_encoders['district'].transform([district])[0]
        soil_enc = label_encoders['soil_type'].transform([soil_type])[0]
        
        input_data = [
            district_enc, soil_enc, ph_value, nitrogen,
            phosphorous, potassium, rainfall, temperature, farm_size
        ]
        
        crop_enc = crop_model.predict([input_data])[0]
        crop = label_encoders['crop'].inverse_transform([crop_enc])[0]
        
        st.success(f"### Recommended Crop: **{crop}**")
        st.write("All possible crops:", ", ".join(label_encoders['crop'].classes_))

elif page == "Agri Chatbot":
    st.header('Agricultural Chatbot Assistant')
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me about agriculture..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        messages = [
            {"role": "system", "content": "You are an Agricultural Chatbot"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = g4f.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.6,
                top_p=0.9
            )
            
            with st.chat_message("assistant"):
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception as e:
            st.error(f"Error generating response: {e}")