import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load('crop_recommender_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, label_encoders

model, label_encoders = load_model()

# App title
st.title('🌱 Smart Crop Recommendation System')
st.markdown("""
This app recommends the most suitable crop based on your agricultural conditions.
""")

# Sidebar with info
st.sidebar.header('About')
st.sidebar.info("""
This system uses machine learning to analyze soil and weather conditions 
to recommend optimal crops for cultivation.
""")

# Input form
with st.form("crop_recommendation_form"):
    st.header('Enter Agricultural Parameters')
    
    col1, col2 = st.columns(2)
    
    with col1:
        district = st.selectbox(
            'District',
            options=label_encoders['district'].classes_
        )
        soil_type = st.selectbox(
            'Soil Type',
            options=label_encoders['soil_type'].classes_
        )
        ph_value = st.slider(
            'pH Value',
            min_value=0.0, max_value=14.0, value=6.5, step=0.1
        )
        nitrogen = st.number_input(
            'Nitrogen Level (kg/ha)',
            min_value=0, max_value=1000, value=250
        )
        
    with col2:
        phosphorous = st.number_input(
            'Phosphorous Level (kg/ha)',
            min_value=0, max_value=1000, value=35
        )
        potassium = st.number_input(
            'Potassium Level (kg/ha)',
            min_value=0, max_value=1000, value=300
        )
        rainfall = st.number_input(
            'Rainfall (mm)',
            min_value=0, max_value=5000, value=650
        )
        temperature = st.number_input(
            'Temperature (°C)',
            min_value=0, max_value=50, value=28
        )
    
    farm_size = st.slider(
        'Farm Size (hectares)',
        min_value=0.1, max_value=50.0, value=3.5, step=0.1
    )
    
    submit_button = st.form_submit_button("Get Crop Recommendation")

# Prediction function
def recommend_crop(input_data):
    # Encode inputs
    input_data[0] = label_encoders['district'].transform([input_data[0]])[0]
    input_data[1] = label_encoders['soil_type'].transform([input_data[1]])[0]
    
    # Predict
    crop_enc = model.predict([input_data])[0]
    crop = label_encoders['crop'].inverse_transform([crop_enc])[0]
    return crop

# Display results
if submit_button:
    input_data = [
        district,
        soil_type,
        ph_value,
        nitrogen,
        phosphorous,
        potassium,
        rainfall,
        temperature,
        farm_size
    ]
    
    recommended_crop = recommend_crop(input_data)
    
    st.success(f"### Recommended Crop: **{recommended_crop}**")
    
    # Show all possible crops (optional)
    st.subheader("All Possible Crops")
    st.write(", ".join(label_encoders['crop'].classes_))

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)