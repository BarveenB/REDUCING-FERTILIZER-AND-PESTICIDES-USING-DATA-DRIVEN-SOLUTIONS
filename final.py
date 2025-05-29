# This MUST be the first line in your script
import streamlit as st
st.set_page_config(page_title="AgriSmart Assistant", layout="wide")

# Now import other libraries
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from gtts import gTTS
import base64
import g4f

try:
    from tensorflow.keras.preprocessing import image
    from keras.models import load_model
    TENSORFLOW_ENABLED = True
except ImportError:
    st.warning("TensorFlow not available - pest detection will be disabled")
    TENSORFLOW_ENABLED = False

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

crop_model, label_encoders = load_crop_models()

# Load pest detection model if available
if TENSORFLOW_ENABLED:
    @st.cache_resource
    def load_pest_model():
        try:
            model = load_model('model_inception.h5')
            classes = ['rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot', 
                      'asiatic rice borer', 'yellow rice borer', 'rice gall midge', 
                      'Rice Stemfly', 'brown plant hopper', 'white backed plant hopper', 
                      'small brown plant hopper']
            return model, classes
        except Exception as e:
            st.warning(f"Pest detection model not available: {str(e)}")
            return None, None
    
    pest_model, pest_classes = load_pest_model()
else:
    pest_model, pest_classes = None, None

# Fertilizer recommendations
ferti = {
    'Rice Leaf Roller': {
        'en': 'Applying a balanced fertilizer with a high potassium content can help prevent bacterial blight in plants.',
        'ta': 'உயர் பொட்டாசியம் கொண்ட சமநிலை உரம் பயன்படுத்துவதன் மூலம் செடிகளில் பாக்டீரியல் பிளைட் ஏற்படுவதைத் தடுப்பது சாத்தியமாகும்.'
    },
    'Rice Leaf Caterpillar': {
        'en': 'Increasing the nitrogen content in the soil through the application of a nitrogen-rich fertilizer can help prevent blast disease in plants.',
        'ta': 'நைட்ரஜன் நிறைந்த உரம் பயன்படுத்துவதன் மூலம் மண்ணில் நைட்ரஜன் அளவை அதிகரிப்பது செடிகளில் பிளாஸ்ட் நோயைத் தடுப்பதற்கு உதவும்.'
    },
    # Add all other pest recommendations here
}

# App title and description
st.title('🌱 AgriSmart Assistant')
st.markdown("""
This intelligent assistant helps farmers with:
- Crop recommendations based on soil and weather conditions
- Pest detection from plant images (when available)
- Agricultural advice through chatbot
""")

# Navigation
pages = ["Crop Recommendation", "Agri Chatbot"]
if TENSORFLOW_ENABLED and pest_model is not None:
    pages.insert(1, "Pest Detection")

page = st.sidebar.radio("Navigation", pages)

# Crop Recommendation Page
if page == "Crop Recommendation":
    st.header('Crop Recommendation System')
    
    if crop_model is None or label_encoders is None:
        st.error("Crop recommendation models failed to load. Please check the model files.")
    else:
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
            try:
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
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Pest Detection Page
elif page == "Pest Detection":
    st.header('Pest Detection System')
    
    if pest_model is None:
        st.warning("Pest detection model is not available")
    else:
        uploaded_file = st.file_uploader("Upload an image of affected plant", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Create temp directory if needed
            os.makedirs("temp", exist_ok=True)
            temp_file = os.path.join("temp", uploaded_file.name)
            
            try:
                # Save the uploaded file
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Display the image
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                
                if st.button('Detect Pest'):
                    try:
                        # Process the image
                        img_ = image.load_img(temp_file, target_size=(224, 224, 3))
                        img_array = image.img_to_array(img_)
                        img_processed = np.expand_dims(img_array, axis=0)
                        img_processed /= 255.
                        
                        # Make prediction
                        prediction = pest_model.predict(img_processed)
                        index = np.argmax(prediction)
                        result = "Unknown"
                        fer_en = ""
                        fer_ta = ""

                        if index < len(pest_classes):
                            result = str(pest_classes[index]).title()
                            fertilizer_data = ferti.get(result, {})
                            fer_en = fertilizer_data.get('en', "No fertilizer information available.")
                            fer_ta = fertilizer_data.get('ta', "பரிந்துரை கிடைக்கவில்லை.")

                            # Display results
                            st.warning(f"Detected Pest: **{result}**")
                            st.info(f"Recommendation: {fer_en}")
                            
                            # Generate audio
                            try:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**English Audio**")
                                    tts_en = gTTS(text=f"Detected pest: {result}. Recommendation: {fer_en}", lang='en')
                                    tts_en.save("temp/pest_en.mp3")
                                    audio_file = open("temp/pest_en.mp3", "rb")
                                    audio_bytes = audio_file.read()
                                    st.audio(audio_bytes, format='audio/mp3')
                                
                                with col2:
                                    st.markdown("**Tamil Audio**")
                                    tts_ta = gTTS(text=f"கண்டறியப்பட்ட பூச்சி: {result}. பரிந்துரை: {fer_ta}", lang='ta')
                                    tts_ta.save("temp/pest_ta.mp3")
                                    audio_file = open("temp/pest_ta.mp3", "rb")
                                    audio_bytes = audio_file.read()
                                    st.audio(audio_bytes, format='audio/mp3')
                            except Exception as e:
                                st.warning(f"Audio generation failed: {str(e)}")
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
            except Exception as e:
                st.error(f"Error handling uploaded file: {str(e)}")

# Chatbot Page
elif page == "Agri Chatbot":
    st.header('Agricultural Chatbot Assistant')
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask me about agriculture..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an Agricultural Chatbot designed to assist farmers with farming practices, "
                    "pest control, soil management, and government schemes. "
                    "Keep responses focused on agriculture."
                )
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = g4f.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.6,
                top_p=0.9
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")