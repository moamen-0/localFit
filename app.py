import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
from flask import Flask, Response, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import pygame
import time
import base64
import threading
import uuid
from gtts import gTTS
import mediapipe as mp
from utils import calculate_angle, mp_pose, pose
from exercises.bicep_curl import process_frame  # Modified function we'll create

# Initialize Flask app with CORS support
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Configure Socket.IO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Initialize pygame mixer for audio feedback
pygame.mixer.init()

# User session storage
user_sessions = {}

# Audio feedback messages
AUDIO_MESSAGES = {
    "left_arm_forward": "Keep your left arm closer to your body",
    "right_arm_forward": "Keep your right arm closer to your body",
    "both_arms_forward": "Keep both arms closer to your body",
    "left_shoulder_high": "Lower your left shoulder",
    "right_shoulder_high": "Lower your right shoulder",
    "both_shoulders_high": "Lower both shoulders",
    "left_elbow_straight": "Bend your left elbow more",
    "right_elbow_straight": "Bend your right elbow more",
    "both_elbows_straight": "Bend both elbows more"
}

# Create audio directory and files if they don't exist
def setup_audio():
    os.makedirs("audio", exist_ok=True)
    sound_objects = {}
    
    try:
        for key, message in AUDIO_MESSAGES.items():
            filepath = f"audio/{key}.mp3"
            
            # Create audio file if it doesn't exist
            if not os.path.exists(filepath):
                print(f"Creating audio file: {filepath}")
                tts = gTTS(text=message, lang='en')
                tts.save(filepath)
            
            # Load sound object
            sound_objects[key] = pygame.mixer.Sound(filepath)
        return sound_objects
    except Exception as e:
        print(f"Error with audio setup: {e}")
        return {}

# Initialize audio
sound_objects = setup_audio()

# User session class to track exercise state
class UserSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.left_counter = 0
        self.right_counter = 0
        self.left_state = None
        self.right_state = None
        self.current_audio_key = None
        self.current_feedback = ""
        self.last_feedback_time = 0
        self.feedback_cooldown = 2

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = UserSession(session_id)
    emit('session_id', {'session_id': session_id})
    print(f"Client connected: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected")
    # Sessions are cleaned up separately based on session_id

@socketio.on('frame')
def handle_frame(data):
    try:
        # Get the session information
        session_id = data.get('session_id')
        if not session_id or session_id not in user_sessions:
            emit('error', {'message': 'Invalid session ID'})
            return

        session = user_sessions[session_id]
        
        # Decode the base64 image
        encoded_data = data.get('image')
        if not encoded_data:
            emit('error', {'message': 'No image data received'})
            return
            
        # Strip the prefix (e.g., 'data:image/jpeg;base64,')
        if ',' in encoded_data:
            encoded_data = encoded_data.split(',')[1]
            
        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('error', {'message': 'Failed to decode image'})
            return
            
        # Process the frame
        processed_frame, results = process_frame(
            frame, 
            session.left_counter, 
            session.right_counter,
            session.left_state,
            session.right_state,
            session.current_audio_key,
            session.current_feedback,
            session.last_feedback_time,
            sound_objects
        )
        
        # Update session data
        session.left_counter = results['left_counter']
        session.right_counter = results['right_counter']
        session.left_state = results['left_state']
        session.right_state = results['right_state']
        session.current_audio_key = results['current_audio_key']
        session.current_feedback = results['current_feedback']
        session.last_feedback_time = results['last_feedback_time']
        
        # Encode the processed image to send back
        _, buffer = cv2.imencode('.jpg', processed_frame)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        
        # Send feedback to client
        emit('feedback', {
            'image': f'data:image/jpeg;base64,{encoded_img}',
            'left_counter': session.left_counter,
            'right_counter': session.right_counter,
            'feedback': session.current_feedback,
            'audio_key': session.current_audio_key
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('error', {'message': f'Error processing frame: {str(e)}'})

# RESTful API endpoint
@app.route('/api/start_session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = UserSession(session_id)
    return jsonify({'session_id': session_id})

@app.route('/api/end_session/<session_id>', methods=['POST'])
def end_session(session_id):
    if session_id in user_sessions:
        del user_sessions[session_id]
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Session not found'}), 404

@app.route('/api/audio/<audio_key>', methods=['GET'])
def get_audio(audio_key):
    if audio_key in AUDIO_MESSAGES:
        filepath = f"audio/{audio_key}.mp3"
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                audio_data = f.read()
            return Response(audio_data, mimetype='audio/mpeg')
    return jsonify({'status': 'error', 'message': 'Audio not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """For Google Cloud health checks"""
    return jsonify({'status': 'healthy'}), 200

# Session cleanup thread
def cleanup_sessions():
    """Remove inactive sessions periodically"""
    while True:
        # Sleep for 1 hour before checking
        time.sleep(3600)
        # For a more sophisticated implementation, track last activity time
        # and remove sessions inactive for more than X time

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    # Get port from environment variable for Cloud compatibility
    port = int(os.environ.get('PORT', 8080))
    
    # In production, use gevent or another production-ready server
    socketio.run(app, host='0.0.0.0', port=port, debug=False)