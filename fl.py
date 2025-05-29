import streamlit as st
import cv2
import numpy as np
import requests
import datetime
import joblib
import g4f
from PIL import Image

# ----------- Custom CSS for Styling -----------
st.markdown("""
    <style>
    .main {
        background-color: #f0f7f0;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput, .stSelectbox, .stSlider {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 5px;
    }
    .stSpinner {
        color: #4CAF50;
    }
    h1, h2, h3 {
        color: #2e7d32;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
    }
    .recommendation-box {
        background-color: #ffffff;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 20px;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- Load ML Model for Soil Analysis -----------
@st.cache_resource
def load_soil_model():
    return joblib.load("soil_model.pkl")  # Dummy model

# ----------- Leaf Color Index Detection -----------
def get_leaf_color_index():
    cap = cv2.VideoCapture(0)
    st.info("📸 Press 's' to capture the leaf image.")

    leaf_index = 3  # default fallback
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access camera.")
            break

        cv2.imshow("Leaf Capture - Press 's'", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_pixels = cv2.bitwise_and(frame, frame, mask=mask)
            mean_val = cv2.mean(green_pixels, mask=mask)
            green_intensity = mean_val[1]

            if green_intensity < 50:
                leaf_index = 1
            elif green_intensity < 100:
                leaf_index = 2
            elif green_intensity < 150:
                leaf_index = 3
            elif green_intensity < 200:
                leaf_index = 4
            else:
                leaf_index = 5
            break

    cap.release()
    cv2.destroyAllWindows()
    return leaf_index

# ----------- Weather API with Error Handling -----------
def get_weather_forecast(city):
    api_key = "5b0d498b57a2899ac882b7f6b8544290"  # Hardcoded API key
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if "list" not in data:
        raise Exception(f"API error: {data.get('message', 'Unknown error')}")

    forecast = {}
    for item in data['list']:
        date = item['dt_txt'].split()[0]
        rainfall = item.get("rain", {}).get("3h", 0)
        forecast[date] = forecast.get(date, 0) + rainfall

    today = datetime.date.today()
    next_3_days = [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3)]

    weather_data = []
    for d in next_3_days:
        weather_data.append({
            "date": d,
            "rainfall_mm": round(forecast.get(d, 0), 2)
        })
    return weather_data

# ----------- GenAI Recommendation -----------
def get_fertilizer_advice(soil, leaf_index, weather_data, crop, region):
    prompt = f"""
You are an expert agricultural advisor.
Analyze the following data and recommend if fertilizer should be applied or not.

Soil Test:
- pH: {soil['pH']}
- Organic Matter: {soil['organic_matter']}%
- Nitrogen Level: {soil['nitrogen_level']}
- Microbial Activity: {soil['microbial_activity']}

Leaf Color Index (1-5): {leaf_index}

Crop: {crop}
Region: {region}

Rainfall forecast (next 3 days):\n""" + \
    "\n".join([f"- {d['date']}: {d['rainfall_mm']} mm" for d in weather_data]) + \
    """

Provide clear fertilizer advice in **both English and Tamil**. For each language, include:
1. Whether fertilizer should be applied or postponed.
2. Type of fertilizer needed.
3. Quantity and usage method.
4. Simple tips for Indian farmers.

Format the response clearly, separating the English and Tamil sections with headers.
"""

    messages = [
        {"role": "system", "content": "You are a helpful and smart agriculture expert."},
        {"role": "user", "content": prompt}
    ]

    response = g4f.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        top_p=0.9
    )
    return response

# ----------- Streamlit UI -----------
st.title("🌱 Smart Fertilizer Recommendation System")
st.markdown("Empowering farmers with AI-driven fertilizer advice in English and Tamil.")

# Header Image (Placeholder - replace with a real image URL or local file)
st.image("https://via.placeholder.com/800x200.png?text=Agriculture+Banner", use_column_width=True)

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    city = st.text_input("Enter City for Weather", value="Trichy", help="Enter your city for accurate weather forecasts.")
    crop = st.selectbox("Select Crop", ["Rice", "Wheat", "Maize", "Sugarcane"], help="Choose the crop you are growing.")
    region = st.text_input("Region", value="Tamil Nadu", help="Enter your state or region.")
    use_camera = st.checkbox("Use Webcam for Leaf Color", value=False, help="Enable to capture leaf color via webcam.")

# Soil Test Data Input
st.header("📋 Soil Test Data")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        pH = st.slider("Soil pH", 4.0, 9.0, 6.5, help="Typical range: 6.0-7.5 for most crops.")
        organic_matter = st.slider("Organic Matter (%)", 0.0, 10.0, 2.5, help="Higher values indicate fertile soil.")
    with col2:
        nitrogen_level = st.selectbox("Nitrogen Level", ["low", "medium", "high"], help="Affects plant growth and yield.")
        microbial_activity = st.selectbox("Microbial Activity", ["low", "moderate", "high"], help="Indicates soil health.")

# Recommendation Button
if st.button("🧠 Generate Recommendation", use_container_width=True):
    soil = {
        "pH": pH,
        "organic_matter": organic_matter,
        "nitrogen_level": nitrogen_level,
        "microbial_activity": microbial_activity
    }

    # Progress Bar
    progress = st.progress(0)
    
    # Leaf Color Analysis
    with st.spinner("📸 Analyzing leaf color..."):
        leaf_index = get_leaf_color_index() if use_camera else 3
        st.success(f"Leaf Color Index: {leaf_index}/5")
        progress.progress(33)

    # Weather Forecast
    with st.spinner("🌤 Fetching weather forecast..."):
        try:
            weather = get_weather_forecast(city)
            with st.expander("Weather Forecast"):
                for day in weather:
                    st.write(f"**{day['date']}**: {day['rainfall_mm']} mm")
        except Exception as e:
            st.warning(f"⚠️ Weather API failed: {e}")
            weather = [
                {"date": "Today", "rainfall_mm": 0},
                {"date": "Tomorrow", "rainfall_mm": 0},
                {"date": "Day After", "rainfall_mm": 0},
            ]
        progress.progress(66)

    # GenAI Recommendation
    with st.spinner("🤖 Generating recommendation..."):
        try:
            result = get_fertilizer_advice(soil, leaf_index, weather, crop, region)
            progress.progress(100)
            with st.container():
                st.subheader("📊 Fertilizer Recommendation")
                st.markdown(f"<div class='recommendation-box'>{result}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ GPT error: {e}")
            progress.progress(100)

# Footer
st.markdown("<div class='footer'>Developed with ❤️ by xAI | Powered by OpenWeatherMap and GPT-4o-mini</div>", unsafe_allow_html=True)