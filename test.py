from __future__ import division, print_function
import cv2
import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import filedialog
from tkinter import Tk
from teachable_machine import TeachableMachine
from tensorflow.keras.models import load_model
model = load_model('model_inception.h5',compile=False)

#from mail import report_send_mail
#from mail import*


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#MODEL_PATH = 'keras_model.h5'
model = load_model('model_inception.h5',compile=False)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    index = np.argmax(preds)
   # class_name = class_names[index]
    confidence_score = preds[index]
    print("confidence_score:",confidence_score)
    #Print prediction and confidence score
   # print("Class:", class_name[2:], end="")
    
    
    if preds == 1:
        preds = "rice leaf roller"
        print("rice leaf roller")
        #ser.write(b'1')
        #print(ser.write(b'1'))
        #report_send_mail(preds, 'image.jpg')
        #time.sleep(3)
    elif preds==2:
        preds = " rice leaf caterpillar "
        print(" rice leaf caterpillar ")
        #ser.write(b'2')
        #report_send_mail(preds, 'image.jpg')
        #print(ser.write(b'2'))
        #time.sleep(3)
    elif preds==3:
        preds = " paddy stem maggot "
        print(" paddy stem maggot ")
        #ser.write(b'3')
        #report_send_mail(preds, 'image.jpg')
        #print(ser.write(b'3'))
        #time.sleep(3)

    elif preds==4:
        preds = "asiatic rice borer"
        print("asiatic rice borer")
        #ser.write(b'4')
        #report_send_mail(preds, 'image.jpg')
        #print(ser.write(b'4'))
        #time.sleep(3)

    elif preds==5:
        preds = " yellow rice borer "
        print(" yellow rice borer  ")
        #ser.write(b'5')
        #report_send_mail(preds, 'image.jpg')
        #print(ser.write(b'4'))
        #time.sleep(3)
    elif preds==6:
        preds = " rice gall midge "
        print(" rice gall midge ")
        #ser.write(b'6')
        #report_send_mail(preds, 'image.jpg')
        #print(ser.write(b''))
        #time.sleep(3)
    elif preds==7:
        preds = "Rice Stemfly "
        print("Rice Stemfly ")
        #ser.write(b'6')
        #report_send_mail(preds, 'image.jpg')
        #print(ser.write(b''))
        #time.sleep(3)

    elif preds==8:
        preds = " brown plant hopper "

        print(" brown plant hopper ")

        
        #ser.write(b'6')
        #report_send_mail(preds, 'image.jpg')
        #print(ser.write(b''))
        #time.sleep(3)
              
    elif preds==9:
        preds = "white backed plant hopper"
        print("white backed plant hopper ")

        #ser.write(b'6')
        #report_send_mail(preds, 'image.jpg')
        #print(ser.write(b''))
        #time.sleep(3)                

    elif preds==10:
        preds = "small brown plant hopper"
        print("small brown plant hopper")
        #ser.write(b'6')
        #report_send_mail(preds, 'image.jpg')
        #print(ser.write(b''))
        #time.sleep(10)
    return preds



# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the video file path

# Check if the camera/video is opened successfully
if not cap.isOpened():
    print("Error opening video capture.")
    exit()

# Set the video capture duration (in seconds)
capture_duration = 10

# Set the frame rate of the video capture
frame_rate = 30  # Adjust as per your requirement

# Calculate the number of frames to capture
num_frames = int(capture_duration * frame_rate)

# Capture the frames
for i in range(num_frames):
    ret, frame = cap.read()  # Read a frame from the video capture

    if not ret:
        print("Error reading frame.")
        break

    cv2.imshow("Video Capture", frame)  # Display the frame

    # Wait for 1ms and check if the user pressed the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the last captured frame as an image
image_path = "image.jpg"  # Specify the path and filename for the image
cv2.imwrite(image_path, frame)

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()

print("Image saved successfully.")



       

a = "image.jpg"

b=model_predict(a,model)
c = b 
print(c)



