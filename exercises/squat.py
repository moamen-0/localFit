import cv2
import pygame
import os
from gtts import gTTS
from utils import calculate_angle, mp_pose, pose

def squat(sound):
    """
    Track squat exercise
    
    Args:
        sound: Pygame sound object for alerts
        
    Yields:
        Video frames with pose tracking
    """
    counter = 0  # Counter for squats
    state = None  # State for squat position
    cap = cv2.VideoCapture(0)
    voice_playing = False  # Flag to track if voice is playing
    current_instruction = None  # Track current instruction being played
    
    # Initialize pygame mixer if not already initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    
    # Create audio directory if it doesn't exist
    os.makedirs("audio", exist_ok=True)
    
    # Define feedback instructions
    instructions = {
        "knee_angle": "WARNING! KNEE ANGLE TOO LOW! ADJUST YOUR POSITION!",
        "back_straight": "KEEP YOUR BACK STRAIGHT!",
        "feet_position": "KEEP YOUR FEET SHOULDER WIDTH APART!",
        "slow_down": "SLOW DOWN! CONTROL THE MOVEMENT!"
    }
    
    # Dictionary to store voice instruction sound objects
    voice_objects = {}
    
    # Generate voice instructions if needed
    try:
        from gtts import gTTS
        
        for key, message in instructions.items():
            filepath = f"audio/squat_{key}.mp3"
            
            # Create audio file if it doesn't exist
            if not os.path.exists(filepath):
                print(f"Creating voice instruction: {filepath}")
                tts = gTTS(text=message, lang='en')
                tts.save(filepath)
            
            # Load sound object
            voice_objects[key] = pygame.mixer.Sound(filepath)
    except ImportError:
        print("gTTS not available, voice files must be created manually")
    except Exception as e:
        print(f"Error with voice setup: {e}")
    
    print("Squat exercise started")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        form_violated = False
        current_violation = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Define landmarks for both legs
            leg_sides = {
                'left': {
                    'hip': mp_pose.PoseLandmark.LEFT_HIP,
                    'knee': mp_pose.PoseLandmark.LEFT_KNEE,
                    'ankle': mp_pose.PoseLandmark.LEFT_ANKLE
                },
                'right': {
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP,
                    'knee': mp_pose.PoseLandmark.RIGHT_KNEE,
                    'ankle': mp_pose.PoseLandmark.RIGHT_ANKLE
                }
            }
            
            for side, joints in leg_sides.items():
                # Get coordinates for each side
                hip = [
                    landmarks[joints['hip'].value].x,
                    landmarks[joints['hip'].value].y,
                ]
                knee = [
                    landmarks[joints['knee'].value].x,
                    landmarks[joints['knee'].value].y,
                ]
                ankle = [
                    landmarks[joints['ankle'].value].x,
                    landmarks[joints['ankle'].value].y,
                ]
                
                # Convert normalized coordinates to image coordinates
                hip_coords = (int(hip[0] * image.shape[1]), int(hip[1] * image.shape[0]))
                knee_coords = (int(knee[0] * image.shape[1]), int(knee[1] * image.shape[0]))
                ankle_coords = (int(ankle[0] * image.shape[1]), int(ankle[1] * image.shape[0]))
                
                # Draw lines between hip, knee, and ankle
                cv2.line(image, hip_coords, knee_coords, (0, 255, 0), 2)  # Green line
                cv2.line(image, knee_coords, ankle_coords, (0, 255, 0), 2)  # Green line
                
                # Draw circles at hip, knee, and ankle
                cv2.circle(image, hip_coords, 7, (0, 0, 255), -1)  # Red circle
                cv2.circle(image, knee_coords, 7, (0, 0, 255), -1)  # Red circle
                cv2.circle(image, ankle_coords, 7, (0, 0, 255), -1)  # Red circle
                
                # Calculate angles
                knee_angle = calculate_angle(hip, knee, ankle)
                
                # Display angles with default color (white)
                knee_color = (255, 255, 255)
                
                # Check form violations
                if knee_angle < 70:  # Knee angle too low
                    form_violated = True
                    current_violation = "knee_angle"
                    knee_color = (0, 0, 255)  # Red for violation
                
                # Display angles with appropriate colors
                cv2.putText(
                    image,
                    f'K: {int(knee_angle)}°',
                    knee_coords,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    knee_color,
                    2,
                    cv2.LINE_AA
                )
                
                # Track exercise state and count reps
                if knee_angle < 90:
                    state = "down"
                if knee_angle > 160 and state == "down":
                    state = "up"
                    counter += 1
                    print(f'Squat Counter: {counter}')
            
            # Voice feedback control
            if form_violated and current_violation:
                if not voice_playing or current_instruction != current_violation:
                    pygame.mixer.stop()
                    if current_violation in voice_objects:
                        voice_objects[current_violation].play()
                        voice_playing = True
                        current_instruction = current_violation
            elif voice_playing:
                pygame.mixer.stop()
                voice_playing = False
                current_instruction = None
            
            # Display instruction message when form is violated
            if voice_playing and current_instruction:
                text = instructions[current_instruction]
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                text_y = image.shape[0] // 2
                
                # Draw semi-transparent background
                overlay = image.copy()
                cv2.rectangle(overlay, 
                             (text_x - 10, text_y - text_size[1] - 10),
                             (text_x + text_size[0] + 10, text_y + 10),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                
                # Draw text
                cv2.putText(image, text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display current state and counter
            cv2.putText(image, f'State: {state.upper() if state else "START"}', (10, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Counter: {counter}', (10, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Add form status indicator
            form_status = "GOOD FORM" if not form_violated else "FIX YOUR FORM"
            form_color = (0, 255, 0) if not form_violated else (0, 0, 255)
            cv2.putText(image, form_status, (10, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, form_color, 2, cv2.LINE_AA)
            
            # Add exercise guidance
            cv2.putText(image, "Squat Exercise", (10, image.shape[0] - 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Keep knees above 70°", (10, image.shape[0] - 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Keep back straight", (10, image.shape[0] - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Feet shoulder width apart", (10, image.shape[0] - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Convert to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 