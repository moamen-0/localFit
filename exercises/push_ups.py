import cv2
import pygame
import os
from gtts import gTTS
from utils import calculate_angle, mp_pose, pose

def push_ups(sound):
    """
    Track push-ups exercise with voice instructions
    
    Args:
        sound: Pygame sound object for alerts
        
    Yields:
        Video frames with pose tracking
    """
    counter = 0  # Counter for reps
    stage = None  # State of the exercise
    cap = cv2.VideoCapture(0)
    voice_playing = False  # Flag to track if voice is playing
    current_instruction = None  # Track current instruction being played
    
    # Initialize pygame mixer if not already initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    
    # Create audio directory if it doesn't exist
    os.makedirs("audio", exist_ok=True)
    
    # Define clear and helpful instructions for the user
    instructions = {
        "body_straight": "KEEP YOUR BODY STRAIGHT!",
        "elbow_angle": "KEEP YOUR ELBOWS AT 90 DEGREES WHEN DOWN!",
        "full_extension": "FULLY EXTEND YOUR ARMS AT THE TOP!",
        "head_position": "KEEP YOUR HEAD IN LINE WITH YOUR BODY!"
    }
    
    # Dictionary to store voice instruction sound objects
    voice_objects = {}
    
    # Generate voice instructions if needed
    try:
        from gtts import gTTS
        
        for key, message in instructions.items():
            filepath = f"audio/push_ups_{key}.mp3"
            
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
    
    print("Push-ups Exercise Started")
    
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
        instruction_message = ""
        current_violation = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Define landmarks for both arms and body
            arm_sides = {
                'left': {
                    'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                    'hip': mp_pose.PoseLandmark.LEFT_HIP,
                    'knee': mp_pose.PoseLandmark.LEFT_KNEE
                },
                'right': {
                    'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                    'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                    'hip': mp_pose.PoseLandmark.RIGHT_HIP,
                    'knee': mp_pose.PoseLandmark.RIGHT_KNEE
                }
            }
            
            # Calculate overall body angle
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ]
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
            
            # Calculate body midpoint
            body_midpoint_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, 
                                    (left_shoulder[1] + right_shoulder[1])/2]
            body_midpoint_hip = [(left_hip[0] + right_hip[0])/2, 
                                (left_hip[1] + right_hip[1])/2]
            
            # Vertical point above midpoint
            vertical_point = [body_midpoint_shoulder[0], body_midpoint_shoulder[1] - 0.2]
            
            # Body angle
            body_angle = calculate_angle(vertical_point, body_midpoint_shoulder, body_midpoint_hip)
            
            # Variables to track arm states
            left_arm_state = "up"
            right_arm_state = "up"
            
            # Process each arm
            for side, joints in arm_sides.items():
                # Get joint coordinates
                shoulder = [
                    landmarks[joints['shoulder'].value].x,
                    landmarks[joints['shoulder'].value].y
                ]
                elbow = [
                    landmarks[joints['elbow'].value].x,
                    landmarks[joints['elbow'].value].y
                ]
                wrist = [
                    landmarks[joints['wrist'].value].x,
                    landmarks[joints['wrist'].value].y
                ]
                hip = [
                    landmarks[joints['hip'].value].x,
                    landmarks[joints['hip'].value].y
                ]
                
                # Convert to pixel coordinates
                shoulder_coords = (int(shoulder[0] * image.shape[1]), int(shoulder[1] * image.shape[0]))
                elbow_coords = (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0]))
                wrist_coords = (int(wrist[0] * image.shape[1]), int(wrist[1] * image.shape[0]))
                hip_coords = (int(hip[0] * image.shape[1]), int(hip[1] * image.shape[0]))
                
                # Draw lines and points
                cv2.line(image, shoulder_coords, elbow_coords, (0, 255, 0), 2)
                cv2.line(image, elbow_coords, wrist_coords, (0, 255, 0), 2)
                cv2.line(image, shoulder_coords, hip_coords, (0, 255, 0), 2)
                
                cv2.circle(image, shoulder_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, elbow_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, wrist_coords, 7, (0, 0, 255), -1)
                cv2.circle(image, hip_coords, 7, (0, 0, 255), -1)
                
                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(hip, shoulder, elbow)
                
                # Display angles
                elbow_color = (255, 255, 255)  # Default white
                
                # Check form
                if elbow_angle < 130:  # Down position
                    if side == 'left':
                        left_arm_state = "down"
                    else:
                        right_arm_state = "down"
                    
                    # Check body angle
                    if body_angle > 20:  # Body bent more than 20 degrees
                        form_violated = True
                        current_violation = "body_straight"
                        elbow_color = (0, 0, 255)  # Red for violation
                    
                    # Check elbow angle
                    if not (85 <= elbow_angle <= 95):  # Not at 90 degrees
                        form_violated = True
                        current_violation = "elbow_angle"
                        elbow_color = (0, 0, 255)  # Red for violation
                
                elif elbow_angle > 170:  # Up position
                    if side == 'left':
                        left_arm_state = "up"
                    else:
                        right_arm_state = "up"
                    
                    # Check for full extension
                    if elbow_angle < 170:
                        form_violated = True
                        current_violation = "full_extension"
                        elbow_color = (0, 0, 255)  # Red for violation
                
                cv2.putText(
                    image,
                    f'E: {int(elbow_angle)}°',
                    elbow_coords,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    elbow_color,
                    2,
                    cv2.LINE_AA
                )
                
                cv2.putText(
                    image,
                    f'S: {int(shoulder_angle)}°',
                    shoulder_coords,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Count repetitions
            if left_arm_state == "down" and right_arm_state == "down":
                stage = "down"
            elif left_arm_state == "up" and right_arm_state == "up" and stage == "down":
                counter += 1
                stage = "up"
                print(f"Push-up counted! Total: {counter}")
            
            # Handle voice feedback
            if form_violated and current_violation:
                if not voice_playing or current_instruction != current_violation:
                    pygame.mixer.stop()
                    if current_violation in voice_objects:
                        voice_objects[current_violation].play()
                        voice_playing = True
                        current_instruction = current_violation
                        instruction_message = instructions[current_violation]
            elif voice_playing:
                pygame.mixer.stop()
                voice_playing = False
                current_instruction = None
            
            # Display instruction message
            if form_violated and instruction_message:
                text_size = cv2.getTextSize(instruction_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
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
                cv2.putText(
                    image,
                    instruction_message,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Display counter and stage
            cv2.putText(
                image,
                f'Count: {counter}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
            
            cv2.putText(
                image,
                f'Stage: {stage if stage else "None"}',
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
            
            # Add form status indicator
            form_status = "GOOD FORM" if not form_violated else "FIX YOUR FORM"
            form_color = (0, 255, 0) if not form_violated else (0, 0, 255)
            
            cv2.putText(
                image,
                form_status,
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                form_color,
                2,
                cv2.LINE_AA
            )
            
            # Add exercise guidance
            cv2.putText(
                image,
                "Keep body straight",
                (10, image.shape[0] - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                image,
                "Elbows at 90° when down",
                (10, image.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                image,
                "Full extension at top",
                (10, image.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Convert image to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # Yield frame for Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 