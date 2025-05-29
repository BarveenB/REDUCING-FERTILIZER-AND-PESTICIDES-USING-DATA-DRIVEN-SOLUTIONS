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
import matplotlib.pyplot as plt

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
    pest_model_path = r"E:\soundarya\crop recommentation\pest detection audio\pest detection audio\shufflenet_model.pth"
    crop_model_path = 'crop_recommender_model.pkl'
    label_encoders_path = 'label_encoders.pkl'

# Pest Classifier Class
class PestClassifier:
    def __init__(self):
        self.classes = Config.pest_classes
        self.device = Config.device
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self):
        model = models.shufflenet_v2_x1_0(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        
        try:
            model.load_state_dict(torch.load(Config.pest_model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading pest model: {str(e)}")
            return None

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image):
        try:
            img_t = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_t)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            
            return {
                'class': self.classes[predicted.item()],
                'confidence': probabilities[predicted.item()].item(),
                'all_predictions': {
                    cls_name: prob.item() 
                    for cls_name, prob in zip(self.classes, probabilities)
                }
            }
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return None

# Load crop recommendation models
@st.cache_resource
def load_crop_models():
    try:
        crop_model = joblib.load(Config.crop_model_path)
        label_encoders = joblib.load(Config.label_encoders_path)
        return crop_model, label_encoders
    except Exception as e:
        st.error(f"Failed to load crop models: {str(e)}")
        return None, None

# Initialize models
crop_model, label_encoders = load_crop_models()
pest_classifier = PestClassifier()

# Fertilizer recommendations
ferti = {
    'rice leaf roller': {
        'en': 'Apply neem oil spray (5ml/liter) weekly. Use balanced NPK fertilizer (10-10-10).',
        'ta': 'வாரம் ஒருமுறை வேப்ப எண்ணெய் தெளிப்பு (5ml/லிட்டர்). சமநிலை NPK உரம் (10-10-10).'
    },
    'rice leaf caterpillar': {
        'en': 'Use chlorantraniliprole spray. Apply nitrogen-rich fertilizer.',
        'ta': 'குளோராண்ட்ரானிலிப்ரோல் தெளிப்பு. நைட்ரஜன் நிறைந்த உரம்.'
    },
    # Add all other pests...
    'default': {
        'en': 'Consult local agricultural extension officer for specific recommendations.',
        'ta': 'குறிப்பிட்ட பரிந்துரைகளுக்கு உள்ளூர் வேளாண்மை அலுவலரை அணுகவும்.'
    }
}

# App UI
st.title('🌱 AgriSmart Assistant')
st.markdown("""
**Farmer's Intelligent Assistant**  
Crop recommendations | Pest detection | Expert advice
""")

# Navigation
pages = ["Crop Recommendation", "Agri Chatbot"]
if pest_classifier.model is not None:
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
                    os.remove('rec.mp3')
                except Exception as e:
                    st.warning(f"Audio generation failed: {str(e)}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Pest Detection
elif page == "Pest Detection" and pest_classifier.model is not None:
    st.header('🐛 Pest Identification')
    
    uploaded_file = st.file_uploader("Upload plant image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Classify Pest"):
            with st.spinner("Analyzing..."):
                result = pest_classifier.predict(image)
            
            if result:
                pest = result['class']
                confidence = result['confidence']
                
                st.warning(f"**Detected Pest:** {pest} (Confidence: {confidence:.2f}%)")
                
                # Show detailed predictions
                st.subheader("Detailed Predictions:")
                for class_name, conf in result['all_predictions'].items():
                    st.progress(int(conf), text=f"{class_name}: {conf:.2f}%")
                
                # Visualization
                fig, ax = plt.subplots()
                classes = list(result['all_predictions'].keys())
                confidences = list(result['all_predictions'].values())
                ax.barh(classes, confidences)
                ax.set_xlabel('Confidence (%)')
                ax.set_title('Prediction Confidence Levels')
                st.pyplot(fig)
                
                # Show recommendation
                lang = st.radio("Language", ['English', 'Tamil'])
                lang_key = 'en' if lang == 'English' else 'ta'
                recommendation = ferti.get(pest.lower(), ferti['default']).get(lang_key, "No recommendation available")
                st.info(f"**Treatment:** {recommendation}")
                
                # Audio output
                if st.checkbox("Enable audio feedback", True):
                    audio_text = f"Detected {pest}. Recommendation: {recommendation}"
                    try:
                        tts = gTTS(text=audio_text, lang=lang_key)
                        tts.save('pest.mp3')
                        st.audio('pest.mp3')
                        os.remove('pest.mp3')
                    except Exception as e:
                        st.warning(f"Audio generation failed: {str(e)}")

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