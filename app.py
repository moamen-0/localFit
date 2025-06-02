import eventlet
eventlet.monkey_patch()

import os
os.environ['SDL_AUDIODRIVER'] = 'dummy'
from flask import Flask, Response, request, jsonify, send_from_directory
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
import logging
import queue
from exercises.front_raise import dumbbell_front_raise
from exercises.side_lateral_raise import side_lateral_raise
from exercises.triceps_kickback import triceps_kickback
from exercises.squat import squat
from exercises.shoulder_press import shoulder_press
from exercises.push_ups import push_ups

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with CORS support
from flask_cors import CORS
app = Flask(__name__, static_folder='static')
CORS(app)

# Configure Socket.IO with optimized settings
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='eventlet',
                   engineio_logger=False,  # Reduce logging
                   logger=False,
                   ping_interval=25,
                   ping_timeout=60,
                   max_http_buffer_size=5 * 1024 * 1024,  # 5MB buffer
                   http_compression=True)

# Initialize pygame mixer for audio feedback
pygame.mixer.init()

# User session storage
user_sessions = {}
active_sessions = {}

# Processing queues for frame handling
frame_queues = {}
MAX_QUEUE_SIZE = 3  # Limit queue size to prevent memory issues

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
                logger.info(f"Creating audio file: {filepath}")
                tts = gTTS(text=message, lang='en')
                tts.save(filepath)
            
            # Load sound object
            sound_objects[key] = pygame.mixer.Sound(filepath)
        return sound_objects
    except Exception as e:
        logger.error(f"Error with audio setup: {e}")
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
        self.last_active = time.time()
        self.frame_count = 0
        self.current_exercise = "bicep_curl"  # Default
        self.processing = False
        self.last_response_time = 0
        self.selected_exercise = None
        self.is_active = True

# Process frame in worker thread to avoid blocking
def process_frame_worker(session_id):
    """
    Worker function to process frames for a session
    """
    if session_id not in user_sessions or session_id not in frame_queues:
        return
        
    session = user_sessions[session_id]
    q = frame_queues[session_id]
    
    logger.info(f"Starting worker thread for session {session_id}")
    
    while True:
        try:
            # Get frame data with timeout to allow thread termination
            try:
                frame_data = q.get(timeout=0.5)
            except queue.Empty:
                # Check if session is still active
                if session_id not in active_sessions:
                    logger.info(f"Worker thread for {session_id} terminating due to inactive session")
                    break
                continue
                
            # Process the frame
            try:
                # Extract image data
                encoded_data = frame_data.get('image')
                if not encoded_data:
                    q.task_done()
                    continue
                    
                # Strip the prefix if present
                if ',' in encoded_data:
                    encoded_data = encoded_data.split(',')[1]
                    
                # Decode image
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    q.task_done()
                    continue
                
                # Import the proper process_frame function based on exercise type
                if session.current_exercise == "front_raise":
                    # Use the front raise exercise generator
                    for processed_frame in dumbbell_front_raise(sound_objects):
                        # Extract the frame from the generator
                        if isinstance(processed_frame, tuple):
                            frame_data = processed_frame[0]
                            if frame_data.startswith(b'--frame'):
                                # Extract the actual image data
                                frame_data = frame_data.split(b'\r\n\r\n')[1]
                                frame_data = frame_data.split(b'\r\n')[0]
                                nparr = np.frombuffer(frame_data, np.uint8)
                                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                break
                elif session.current_exercise == "side_lateral_raise":
                    # Use the side lateral raise exercise generator
                    for processed_frame in side_lateral_raise(sound_objects):
                        # Extract the frame from the generator
                        if isinstance(processed_frame, tuple):
                            frame_data = processed_frame[0]
                            if frame_data.startswith(b'--frame'):
                                # Extract the actual image data
                                frame_data = frame_data.split(b'\r\n\r\n')[1]
                                frame_data = frame_data.split(b'\r\n')[0]
                                nparr = np.frombuffer(frame_data, np.uint8)
                                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                break
                elif session.current_exercise == "triceps_kickback":
                    # Use the triceps kickback exercise generator
                    for processed_frame in triceps_kickback(sound_objects):
                        # Extract the frame from the generator
                        if isinstance(processed_frame, tuple):
                            frame_data = processed_frame[0]
                            if frame_data.startswith(b'--frame'):
                                # Extract the actual image data
                                frame_data = frame_data.split(b'\r\n\r\n')[1]
                                frame_data = frame_data.split(b'\r\n')[0]
                                nparr = np.frombuffer(frame_data, np.uint8)
                                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                break
                elif session.current_exercise == "squat":
                    # Use the squat exercise generator
                    for processed_frame in squat(sound_objects):
                        # Extract the frame from the generator
                        if isinstance(processed_frame, tuple):
                            frame_data = processed_frame[0]
                            if frame_data.startswith(b'--frame'):
                                # Extract the actual image data
                                frame_data = frame_data.split(b'\r\n\r\n')[1]
                                frame_data = frame_data.split(b'\r\n')[0]
                                nparr = np.frombuffer(frame_data, np.uint8)
                                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                break
                elif session.current_exercise == "shoulder_press":
                    # Use the shoulder press exercise generator
                    for processed_frame in shoulder_press(sound_objects):
                        # Extract the frame from the generator
                        if isinstance(processed_frame, tuple):
                            frame_data = processed_frame[0]
                            if frame_data.startswith(b'--frame'):
                                # Extract the actual image data
                                frame_data = frame_data.split(b'\r\n\r\n')[1]
                                frame_data = frame_data.split(b'\r\n')[0]
                                nparr = np.frombuffer(frame_data, np.uint8)
                                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                break
                else:
                    # Use the bicep curl exercise
                    from exercises.bicep_curl import process_frame
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
                
                # Update session data for bicep curl
                if session.current_exercise != "front_raise":
                    session.left_counter = results['left_counter']
                    session.right_counter = results['right_counter']
                    session.left_state = results['left_state']
                    session.right_state = results['right_state']
                    session.current_audio_key = results['current_audio_key']
                    session.current_feedback = results['current_feedback']
                    session.last_feedback_time = results['last_feedback_time']
                
                session.frame_count += 1
                
                # Record response time for metrics
                session.last_response_time = time.time()
                
                # Optimize image encoding while keeping quality reasonable
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                encoded_img = base64.b64encode(buffer).decode('utf-8')
                
                # Send feedback to client
                socketio.emit('frame', {
                    'image': f'data:image/jpeg;base64,{encoded_img}',
                    'left_counter': session.left_counter,
                    'right_counter': session.right_counter,
                    'feedback': session.current_feedback,
                    'audio_key': session.current_audio_key
                }, room=session_id)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)
            finally:
                q.task_done()
                
        except Exception as e:
            logger.error(f"Worker thread error: {e}", exc_info=True)
            break

# Add a route for the root path to serve the index.html
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    session_id = request.sid  # Use Socket.IO's session ID
    user_sessions[session_id] = UserSession(session_id)
    active_sessions[session_id] = time.time()
    
    # Create a queue for this session
    frame_queues[session_id] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    
    # Start a worker thread for this session using eventlet.spawn
    eventlet.spawn(process_frame_worker, session_id)
    
    emit('session_id', {'session_id': session_id})
    logger.info(f"Client connected: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    logger.info(f"Client disconnected: {session_id}")
    
    # Clean up resources
    if session_id in user_sessions:
        del user_sessions[session_id]
    if session_id in active_sessions:
        del active_sessions[session_id]
    if session_id in frame_queues:
        # Clear the queue
        try:
            q = frame_queues[session_id]
            while not q.empty():
                q.get_nowait()
                q.task_done()
        except:
            pass
        del frame_queues[session_id]

@socketio.on('select_exercise')
def handle_select_exercise(data):
    """Handle exercise selection from client"""
    if 'exercise' not in data:
        return
    
    exercise = data['exercise']
    print(f"Selected exercise: {exercise}")
    
    # Update session with selected exercise
    if 'user_id' in data:
        user_id = data['user_id']
        if user_id in user_sessions:
            user_sessions[user_id].selected_exercise = exercise
            print(f"Updated exercise for user {user_id}: {exercise}")

@socketio.on('frame')
def handle_frame(data):
    try:
        # Get the session information
        session_id = request.sid
        if session_id not in user_sessions:
            emit('error', {'message': 'Invalid session ID'})
            return

        session = user_sessions[session_id]
        
        # Update last active timestamp
        session.last_active = time.time()
        active_sessions[session_id] = time.time()
        
        # Skip if the processing queue is full (backpressure)
        if session_id in frame_queues:
            q = frame_queues[session_id]
            if q.full():
                # Calculate how many frames we're behind
                backlog = q.qsize()
                
                # Tell client to slow down if seriously behind
                if backlog >= MAX_QUEUE_SIZE:
                    emit('backpressure', {
                        'backlog': backlog,
                        'suggestion': 'reduce_framerate'
                    })
                    return
                
            # Add the frame to the processing queue
            try:
                q.put_nowait(data)
            except queue.Full:
                # Queue is full, drop this frame
                pass
        
    except Exception as e:
        logger.error(f"Error in frame handler: {e}", exc_info=True)
        emit('error', {'message': f'Server error: {str(e)}'})
        
# RESTful API endpoint
@app.route('/api/start_session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = UserSession(session_id)
    active_sessions[session_id] = time.time()
    
    # Create a queue for this session
    frame_queues[session_id] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    
    # Start a worker thread for this session using eventlet.spawn
    eventlet.spawn(process_frame_worker, session_id)
    
    return jsonify({'session_id': session_id})

@app.route('/api/end_session/<session_id>', methods=['POST'])
def end_session(session_id):
    if session_id in user_sessions:
        del user_sessions[session_id]
        if session_id in active_sessions:
            del active_sessions[session_id]
        if session_id in frame_queues:
            del frame_queues[session_id]
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

@app.route('/api/status', methods=['GET'])
def status():
    """API endpoint for monitoring system status"""
    return jsonify({
        'status': 'running',
        'active_sessions': len(active_sessions),
        'total_sessions': len(user_sessions)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """For Google Cloud health checks"""
    return jsonify({'status': 'healthy'}), 200

# Session cleanup thread
def cleanup_sessions():
    """Remove inactive sessions periodically"""
    while True:
        try:
            current_time = time.time()
            inactive_timeout = 300  # 5 minutes
            
            for session_id, last_active in list(active_sessions.items()):
                if current_time - last_active > inactive_timeout:
                    # Remove inactive session
                    if session_id in user_sessions:
                        del user_sessions[session_id]
                    if session_id in active_sessions:
                        del active_sessions[session_id]
                    if session_id in frame_queues:
                        del frame_queues[session_id]
                    logger.info(f"Removed inactive session: {session_id}")
            
            # Log current active sessions count
            if active_sessions:
                logger.info(f"Active sessions: {len(active_sessions)}")
        except Exception as e:
            logger.error(f"Error in session cleanup: {e}")
        
        # Sleep for 60 seconds before next cleanup
        time.sleep(60)

# Start cleanup thread using eventlet.spawn
cleanup_thread = eventlet.spawn(cleanup_sessions)

def process_frames(user_id):
    """Process frames for a user session"""
    if user_id not in user_sessions:
        return
    
    session = user_sessions[user_id]
    frame_queue = session.frame_queue
    
    # Create a sound object for alerts
    sound = pygame.mixer.Sound("audio/alert.mp3")
    
    # Get the appropriate exercise function
    if session.selected_exercise == "bicep_curl":
        exercise_func = bicep_curl
    elif session.selected_exercise == "front_raise":
        exercise_func = dumbbell_front_raise
    elif session.selected_exercise == "side_lateral_raise":
        exercise_func = side_lateral_raise
    elif session.selected_exercise == "triceps_kickback":
        exercise_func = triceps_kickback
    elif session.selected_exercise == "squat":
        exercise_func = squat
    elif session.selected_exercise == "shoulder_press":
        exercise_func = shoulder_press
    elif session.selected_exercise == "push_ups":
        exercise_func = push_ups
    else:
        print(f"Unknown exercise: {session.selected_exercise}")
        return
    
    # Start the exercise tracking
    for frame in exercise_func(sound):
        if not session.is_active:
            break
            
        # Send the frame back to the client
        socketio.emit('exercise_result', {
            'image': frame.decode('utf-8'),
            'user_id': user_id
        })
        
        # Small delay to prevent overwhelming the client
        time.sleep(0.01)

if __name__ == '__main__':
    # Ensure the static folder exists
    os.makedirs('static', exist_ok=True)
    
    # Get port from environment variable for Cloud compatibility
    port = int(os.environ.get('PORT', 8080))
    
    logger.info(f"Starting server on port {port}")
    
    # Run with eventlet
    socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=True)