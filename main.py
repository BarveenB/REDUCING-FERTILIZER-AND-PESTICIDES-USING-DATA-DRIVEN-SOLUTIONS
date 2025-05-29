from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import mysql.connector
import os

mydb = mysql.connector.connect(host="localhost", user="root", password="", database="pest")
mycursor = mydb.cursor()

UPLOAD_FOLDER = 'static/file/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/loginpost', methods=['POST', 'GET'])
def userloginpost():
    global data1
    if request.method == 'POST':
        data1 = request.form.get('uname')
        data2 = request.form.get('password')
        
        print("Username:", data1)  # Debug statement
        print("Password:", data2)  # Debug statement

        if data2 is None:
            return render_template('login.html', msg='Password not provided')

        sql = "SELECT * FROM `users` WHERE `uname` = %s AND `password` = %s"
        val = (data1, data2)

        try:
            mycursor.execute(sql, val)
            account = mycursor.fetchone()  # Fetch one row

            if account:
                # Consume remaining results
                mycursor.fetchall()
                mydb.commit()
                return render_template('index.html')
            else:
                return render_template('login.html', msg='Invalid username or password')
        except mysql.connector.Error as err:
            print("Error:", err)  # Debug statement
            return render_template('login.html', msg='An error occurred. Please try again.')



@app.route('/NewUser')
def newuser():
    return render_template('NewUser2.html')

@app.route('/reg', methods=['POST','GET'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        uname = request.form.get('uname')
        email = request.form.get('email')
        phone = request.form.get('phone')
        age = request.form.get('age')
        password = request.form.get('psw')
        gender = request.form.get('gender')
        sql = "INSERT INTO users (name, uname, email , phone, age, password, gender) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (name, uname, email, phone, age, password, gender)
        mycursor.execute(sql, val)
        mydb.commit()
        return render_template('login.html')
    else:
        return render_template('NewUser2.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        classes = ['rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot', 'asiatic rice borer', 'yellow rice borer', 'rice gall midge', 'Rice Stemfly', 'brown plant hopper', 'white backed plant hopper', 'small brown plant hopper']
        file1 = request.files['filename']
        imgfile = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        
        os.makedirs(os.path.dirname(imgfile), exist_ok=True)
        file1.save(imgfile)
        
        model = load_model('model_inception.h5')
        img_ = image.load_img(imgfile, target_size=(224, 224, 3))
        img_array = image.img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.
        prediction = model.predict(img_processed)
        index = np.argmax(prediction)
        result = "Unknown"
        percentage = 0.0

        if index < len(classes):
            result = str(classes[index]).title()
           
        image_url = url_for('static', filename='file/' + file1.filename)
        return render_template('prediction_result.html', msg=result,  src=imgfile, view='style=display:block', view1='style=display:none')
    elif request.method == 'GET':
        return render_template('index.html')

@app.route('/prediction_result')
def prediction_result():
    result = request.args.get('result')
    image_url = request.args.get('image_url')
    return render_template('prediction_result.html', result=result,  image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True, port=6864)
