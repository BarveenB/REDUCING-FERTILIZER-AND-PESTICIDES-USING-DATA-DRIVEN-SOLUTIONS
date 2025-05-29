# This MUST be the first Streamlit command
import streamlit as st
st.set_page_config(page_title="AgriSmart Assistant", layout="wide")

# Import core libraries
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from gtts import gTTS
import base64
import g4f
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# Configuration
class Config:
    pest_classes = [
        'rice leaf roller', 
        'rice leaf caterpillar', 
        'paddy stem maggot',
        'asiatic rice borer', 
        'yellow rice borer', 
        'rice gall midge',
        'Rice Stemfly', 
        'brown plant hopper', 
        'white backed plant hopper',
        'small brown plant hopper'
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/pest_shufflenet.pth"
    input_size = 224

# PyTorch Pest Detection Model
class PestShuffleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.shufflenet_v2_x1_0(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

# Load crop recommendation models
@st.cache_resource
def load_crop_models():
    try:
        crop_model = joblib.load('crop_recommender_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return crop_model, label_encoders
    except Exception as e:
        st.error(f"Failed to load crop models: {str(e)}")
        return None, None

# Load pest detection model
@st.cache_resource
def load_pest_model():
    try:
        model = PestShuffleNet(len(Config.pest_classes))
        model.load_state_dict(torch.load(Config.model_path, map_location=Config.device))
        model = model.to(Config.device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(Config.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return model, transform
    except Exception as e:
        st.warning(f"Pest detection model not available: {str(e)}")
        return None, None

# Initialize models
crop_model, label_encoders = load_crop_models()
pest_model, pest_transform = load_pest_model()

# Fertilizer recommendations
ferti = {
    'rice leaf roller': {
        'en': 'Apply neem oil spray (5ml/liter) weekly. Use balanced NPK fertilizer (10-10-10).',
        'ta': 'வாரம் ஒருமுறை வேப்ப எண்ணெய் தெளிப்பு (5ml/லிட்டர்). சமநிலை NPK உரம் (10-10-10).'
    },
    # Add recommendations for all classes
}

# App UI
st.title('🌱 AgriSmart Assistant')
st.markdown("""
**Farmer's Intelligent Assistant**  
Crop recommendations | Pest detection | Expert advice
""")

# Navigation
pages = ["Crop Recommendation", "Agri Chatbot"]
if pest_model is not None:
    pages.insert(1, "Pest Detection")

page = st.sidebar.selectbox("Menu", pages)

# Crop Recommendation
if page == "Crop Recommendation":
    st.header('🌾 Smart Crop Recommender')
    
    if crop_model is None or label_encoders is None:
        st.error("Service temporarily unavailable")
    else:
        with st.form("crop_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                district = st.selectbox('District', options=label_encoders['district'].classes_)
                soil_type = st.selectbox('Soil Type', options=label_encoders['soil_type'].classes_)
                ph = st.slider('Soil pH', 3.0, 10.0, 6.5, 0.1)
                nitrogen = st.slider('Nitrogen (kg/ha)', 0, 500, 120)
                
            with col2:
                phosphorus = st.slider('Phosphorus (kg/ha)', 0, 300, 45)
                potassium = st.slider('Potassium (kg/ha)', 0, 600, 200)
                rainfall = st.slider('Annual Rainfall (mm)', 200, 2000, 800)
                temp = st.slider('Avg Temperature (°C)', 10, 40, 28)
            
            submit = st.form_submit_button('Recommend Crop')
        
        if submit:
            try:
                district_enc = label_encoders['district'].transform([district])[0]
                soil_enc = label_encoders['soil_type'].transform([soil_type])[0]
                
                input_data = [[
                    district_enc, soil_enc, ph, nitrogen,
                    phosphorus, potassium, rainfall, temp
                ]]
                
                prediction = crop_model.predict(input_data)[0]
                crop = label_encoders['crop'].inverse_transform([prediction])[0]
                
                st.success(f"### Recommended Crop: **{crop}**")
                
                try:
                    tts = gTTS(f"Recommended crop is {crop}", lang='en')
                    tts.save('rec.mp3')
                    st.audio('rec.mp3')
                except:
                    pass
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Pest Detection
elif page == "Pest Detection" and pest_model is not None:
    st.header('🐛 Pest Identification')
    
    uploaded = st.file_uploader("Upload plant image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption='Uploaded Image', width=300)
        
        if st.button('Analyze'):
            try:
                img_t = pest_transform(img).unsqueeze(0).to(Config.device)
                
                with torch.no_grad():
                    outputs = pest_model(img_t)
                    _, pred = torch.max(outputs, 1)
                    pest = Config.pest_classes[pred.item()]
                
                st.warning(f"**Detected Pest:** {pest}")
                
                # Show recommendation
                lang = st.radio("Language", ['English', 'Tamil'])
                lang_key = 'en' if lang == 'English' else 'ta'
                recommendation = ferti.get(pest.lower(), {}).get(lang_key, "No recommendation available")
                st.info(f"**Treatment:** {recommendation}")
                
                # Audio output
                if st.checkbox("Enable audio feedback", True):
                    audio_text = f"Detected {pest}. Recommendation: {recommendation}"
                    tts = gTTS(text=audio_text, lang=lang_key)
                    tts.save('pest.mp3')
                    st.audio('pest.mp3')

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Chatbot
elif page == "Agri Chatbot":
    st.header('💬 Agri Expert Chat')
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Ask farming questions..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        try:
            response = g4f.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You're an agricultural expert. Provide clear, practical farming advice."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7
            )
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            
        except Exception as e:
            st.error(f"Chat error: {str(e)}")

# Create temp directory
os.makedirs("temp", exist_ok=True)