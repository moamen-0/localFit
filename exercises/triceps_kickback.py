import cv2
import pygame
import os
from gtts import gTTS
from utils import calculate_angle, mp_pose, pose

def triceps_kickback(sound):
    """
    Tracks triceps kickback exercise from a side view
    
    Args:
        sound: Pygame sound object for alerts
        
    Yields:
        Video frames with pose tracking
    """
    counter = 0
    state = "down"
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
        "raise_arm": "RAISE YOUR UPPER ARM! ANGLE TOO LOW!",
        "bend_torso": "BEND TORSO FORWARD PROPERLY!",
        "extend_arm": "EXTEND YOUR ARM FULLY BACKWARD!",
        "slow_down": "SLOW DOWN! CONTROL THE MOVEMENT!"
    }
    
    # Dictionary to store voice instruction sound objects
    voice_objects = {}
    
    # Generate voice instructions if needed
    try:
        from gtts import gTTS
        
        for key, message in instructions.items():
            filepath = f"audio/triceps_kickback_{key}.mp3"
            
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
    
    print("Side View Triceps Kickback exercise started")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # No need to flip frame for side view
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        form_violated = False
        current_violation = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # For side view, we'll focus on the side that's visible to the camera
            # We'll check which shoulder is more visible/confident and use that side
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Determine which side is more visible based on visibility score
            side = 'left' if left_shoulder.visibility > right_shoulder.visibility else 'right'
            
            # Get the landmarks for the selected side
            if side == 'left':
                shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            else:
                shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Convert to coordinate lists for angle calculation
            shoulder_point = [shoulder.x, shoulder.y]
            elbow_point = [elbow.x, elbow.y]
            wrist_point = [wrist.x, wrist.y]
            hip_point = [hip.x, hip.y]
            
            # Convert to pixel coordinates for drawing
            shoulder_coords = (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0]))
            elbow_coords = (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0]))
            wrist_coords = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
            hip_coords = (int(hip.x * image.shape[1]), int(hip.y * image.shape[0]))
            
            # Draw arm lines and connections
            cv2.line(image, shoulder_coords, elbow_coords, (0, 255, 0), 2)
            cv2.line(image, elbow_coords, wrist_coords, (0, 255, 0), 2)
            cv2.line(image, shoulder_coords, hip_coords, (0, 255, 0), 2)
            
            # Draw joint circles
            for point in [shoulder_coords, elbow_coords, wrist_coords, hip_coords]:
                cv2.circle(image, point, 7, (0, 0, 255), -1)
            
            # Calculate angles
            # 1. Elbow angle: between shoulder-elbow-wrist
            elbow_angle = calculate_angle(shoulder_point, elbow_point, wrist_point)
            
            # 2. Upper arm angle: between hip-shoulder-elbow
            upper_arm_angle = calculate_angle(hip_point, shoulder_point, elbow_point)
            
            # 3. Torso angle: between vertical and hip-shoulder line
            vertical_point = [hip_point[0], hip_point[1] - 0.2]  # Point directly above hip
            torso_angle = calculate_angle(vertical_point, hip_point, shoulder_point)
            
            # Display angles with default color (white)
            elbow_color = (255, 255, 255)
            upper_arm_color = (255, 255, 255)
            torso_color = (255, 255, 255)
            
            # Check form violations
            
            # 1. Check if upper arm is at 40 degrees or less
            if upper_arm_angle <= 40:
                form_violated = True
                current_violation = "raise_arm"
                upper_arm_color = (0, 0, 255)  # Red for violation
            
            # 2. Check if torso is properly bent forward (30-60 degrees)
            if torso_angle < 30 or torso_angle > 60:
                form_violated = True
                current_violation = "bend_torso"
                torso_color = (0, 0, 255)  # Red for violation
            
            # Display angles with appropriate colors
            cv2.putText(image, f'E: {int(elbow_angle)}°', elbow_coords, 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, elbow_color, 2)
            cv2.putText(image, f'UA: {int(upper_arm_angle)}°', 
                      (shoulder_coords[0], shoulder_coords[1] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, upper_arm_color, 2)
            cv2.putText(image, f'T: {int(torso_angle)}°', 
                      (hip_coords[0], hip_coords[1] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, torso_color, 2)
            
            # Track exercise state and count reps
            if upper_arm_angle > 40 and 30 <= torso_angle <= 60:
                if elbow_angle < 100 and state == "up":
                    state = "down"
                    print(f"DOWN position detected - Elbow angle: {elbow_angle}")
                elif elbow_angle > 150 and state == "down":
                    state = "up"
                    counter += 1
                    print(f"Rep counted! Total: {counter}")
            
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
            cv2.putText(image, f'State: {state.upper()}', (10, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Counter: {counter}', (10, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Add form status indicator
            form_status = "GOOD FORM" if not form_violated else "FIX YOUR FORM"
            form_color = (0, 255, 0) if not form_violated else (0, 0, 255)
            cv2.putText(image, form_status, (10, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, form_color, 2, cv2.LINE_AA)
            
            # Add exercise guidance
            cv2.putText(image, "Side view - Triceps Kickback", (10, image.shape[0] - 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Bend torso forward 45°", (10, image.shape[0] - 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Keep upper arm ABOVE 40°", (10, image.shape[0] - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Extend arm backward fully", (10, image.shape[0] - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Convert to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 