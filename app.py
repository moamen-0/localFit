from flask import Flask, Response
import cv2
from flask_socketio import SocketIO, emit
import pygame
import time
import os
from gtts import gTTS
#from playsound import playsound
import base64
import threading
from gevent import monkey
# Import exercise modules
from utils import calculate_angle
from exercises.bicep_curl import hummer


app = Flask(__name__)

# Initialize pygame mixer
pygame.mixer.init()
# Load sound
sound = pygame.mixer.Sound(r"D:\\Graduation project\\ai22\\fitness_app 55\\fitness_app 55\\localFit\siren-alert-96052.mp3")

@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    if exercise == 'hummer':
        return Response(hummer(sound), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid exercise", 400

@app.route('/api/pose_data')
def pose_data():
    # Get the pose data and return it as a JSON response
    data = hummer(sound)
    return data

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)