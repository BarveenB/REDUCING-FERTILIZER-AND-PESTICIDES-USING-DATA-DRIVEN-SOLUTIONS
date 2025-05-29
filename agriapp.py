from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from gtts import gTTS
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
DATABASE = 'database.db'
app.secret_key = 'your_secret_key_here'

# -----------------------------
# Database Setup
# -----------------------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

class PestClassifier:
    def __init__(self):
        self.classes = [
            'rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot',
            'asiatic rice borer', 'yellow rice borer', 'rice gall midge',
            'Rice Stemfly', 'brown plant hopper', 'white backed plant hopper',
            'small brown plant hopper'
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self):
        model = models.shufflenet_v2_x1_0(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(torch.load("shufflenet_model.pth", map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_t)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        return {
            'class': self.classes[predicted.item()],
            'confidence': probabilities[predicted.item()].item()
        }

# ---------- Load Models ----------
crop_model = joblib.load('crop_recommender_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
pest_classifier = PestClassifier()

# ---------- Routes ----------
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'danger')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials.', 'danger')

    return render_template('login.html')

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    result = None
    audio_path = None
    if request.method == 'POST':
        try:
            district = request.form['district']
            soil_type = request.form['soil_type']
            season = request.form['season']
            ph = float(request.form['ph'])
            nitrogen = int(request.form['nitrogen'])
            phosphorus = int(request.form['phosphorus'])
            potassium = int(request.form['potassium'])
            rainfall = int(request.form['rainfall'])
            temp = int(request.form['temp'])

            district_enc = label_encoders['district'].transform([district])[0]
            soil_enc = label_encoders['soil_type'].transform([soil_type])[0]
            season_enc = label_encoders['season'].transform([season])[0]

            input_data = [[district_enc, soil_enc, season_enc, ph, nitrogen, phosphorus, potassium, rainfall, temp]]
            prediction = crop_model.predict(input_data)[0]
            crop_name = label_encoders['crop'].inverse_transform([prediction])[0]
            result = crop_name

            tts = gTTS(f"Recommended crop is {crop_name}", lang='en')
            audio_path = os.path.join("static", "crop_audio.mp3")
            tts.save(audio_path)
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('crop.html', result=result, audio_path=audio_path,
                           districts=label_encoders['district'].classes_,
                           soils=label_encoders['soil_type'].classes_,
                           seasons=label_encoders['season'].classes_)

@app.route('/pest', methods=['GET', 'POST'])
def pest():
    pest_result = None
    image_path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            pest_result = pest_classifier.predict(image_path)
    return render_template('pest.html', result=pest_result, image_path=image_path)

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import requests
import datetime
import joblib
import g4f
from g4f.client import Client

client = Client()

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    user_input = ""
    bot_response = ""
    
    if request.method == 'POST':
        user_input = request.form['message']
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_input}],
                web_search=False
            )
            bot_response = response.choices[0].message.content
        except Exception as e:
            bot_response = f"Error: {str(e)}"
    
    return render_template('chatbot.html', user_input=user_input, bot_response=bot_response)

def get_leaf_color_index():
    cap = cv2.VideoCapture(0)
    print("Press 's' to capture the leaf image...")

    leaf_index = 3  # fallback default
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access camera.")
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

def get_weather_forecast(city):
    api_key = "5b0d498b57a2899ac882b7f6b8544290"
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
    """\n\nProvide clear fertilizer advice in **both English and Tamil**. For each language, include:
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

@app.route("/soil", methods=["GET", "POST"])
def sindex():
    result = None
    weather = None
    error = None

    if request.method == "POST":
        city = request.form.get("city")
        crop = request.form.get("crop")
        region = request.form.get("region")
        use_camera = request.form.get("use_camera") == "on"

        try:
            soil = {
                "pH": float(request.form.get("ph")),
                "organic_matter": float(request.form.get("organic_matter")),
                "nitrogen_level": request.form.get("nitrogen_level"),
                "microbial_activity": request.form.get("microbial_activity")
            }

            leaf_index = get_leaf_color_index() if use_camera else 3
            weather = get_weather_forecast(city)
            result = get_fertilizer_advice(soil, leaf_index, weather, crop, region)

        except Exception as e:
            error = str(e)

    return render_template("sindex.html", result=result, weather=weather, error=error)




@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
