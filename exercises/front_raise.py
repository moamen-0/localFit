import cv2
import numpy as np
import time
import os
import pygame
from gtts import gTTS
from utils import calculate_angle, mp_pose, pose

def dumbbell_front_raise(sound):
    """
    Track dumbbell front raise exercise
    
    Args:
        sound: Pygame sound object for alerts (not used with voice feedback)
        
    Yields:
        Video frames with pose tracking
    """
    left_counter = 0
    right_counter = 0
    left_state = "down"
    right_state = "down"
    cap = cv2.VideoCapture(0)
    voice_playing = False  # Flag to track if voice guidance is playing
    current_instruction = None  # Track current instruction
    current_audio_key = None
    current_feedback = None
    last_feedback_time = time.time()
    
    # Initialize pygame mixer if not already initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    
    # Create audio directory if it doesn't exist
    os.makedirs("audio", exist_ok=True)
    
    # Define feedback instructions
    instructions = {
        "lower_arm": "LOWER YOUR ARM! YOUR ANGLE IS TOO HIGH!",
        "elbow_bend": "KEEP YOUR ELBOW SLIGHTLY BENT, NOT LOCKED!",
        "arm_position": "KEEP YOUR ARM IN FRONT OF YOUR BODY!",
        "slow_down": "SLOW DOWN! CONTROL THE MOVEMENT!"
    }
    
    # Dictionary to store voice instruction sound objects
    voice_objects = {}
    
    # Generate voice instructions if needed
    try:
        from gtts import gTTS
        
        for key, message in instructions.items():
            filepath = f"audio/front_raise_{key}.mp3"
            
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        image, results = process_frame(frame, left_counter, right_counter, left_state, right_state, 
                                     current_audio_key, current_feedback, last_feedback_time, voice_objects)

        cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

def process_frame(frame, left_counter, right_counter, left_state, right_state, 
                 current_audio_key, current_feedback, last_feedback_time, sound_objects):
    """
    Process a single frame for front raise exercise
    
    Args:
        frame: The image frame to process
        left_counter: Current count for left arm
        right_counter: Current count for right arm
        left_state: Current state of left arm ('up', 'down', None)
        right_state: Current state of right arm ('up', 'down', None)
        current_audio_key: Current audio feedback key
        current_feedback: Current feedback message
        last_feedback_time: Time of last feedback
        sound_objects: Dictionary of pygame sound objects
        
    Returns:
        processed_frame: The frame with overlays
        results: Dictionary with updated counters and state
    """
    # Process the image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set static image flag for better performance
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    
    # Convert back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Variables for tracking violations
    arm_violated = {'left': False, 'right': False}
    violation_types = {'arm_angle': False, 'elbow': False, 'arm_position': False}
    feedback_cooldown = 2  # Seconds between new audio feedback
    
    # If we detect pose landmarks, process them
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Define landmarks for both arms
        arm_sides = {
            'left': {
                'shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
                'elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
                'wrist': mp_pose.PoseLandmark.LEFT_WRIST,
                'hip': mp_pose.PoseLandmark.LEFT_HIP
            },
            'right': {
                'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                'elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
                'wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
                'hip': mp_pose.PoseLandmark.RIGHT_HIP
            }
        }

        # Get coordinates for both shoulders and both hips
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
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

        # Draw a line between both shoulders
        cv2.line(
            image,
            (int(left_shoulder[0] * image.shape[1]), int(left_shoulder[1] * image.shape[0])),
            (int(right_shoulder[0] * image.shape[1]), int(right_shoulder[1] * image.shape[0])),
            (0, 255, 255),  # Color: yellow
            2
        )

        # Draw a line between both hips
        cv2.line(
            image,
            (int(left_hip[0] * image.shape[1]), int(left_hip[1] * image.shape[0])),
            (int(right_hip[0] * image.shape[1]), int(right_hip[1] * image.shape[0])),
            (0, 255, 255),  # Color: yellow
            2
        )

        for side, joints in arm_sides.items():
            # Get coordinates for each side
            shoulder = [
                landmarks[joints['shoulder'].value].x,
                landmarks[joints['shoulder'].value].y,
            ]
            elbow = [
                landmarks[joints['elbow'].value].x,
                landmarks[joints['elbow'].value].y,
            ]
            wrist = [
                landmarks[joints['wrist'].value].x,
                landmarks[joints['wrist'].value].y,
            ]
            hip = [
                landmarks[joints['hip'].value].x,
                landmarks[joints['hip'].value].y
            ]

            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(hip, shoulder, elbow)

            # Draw connections for the arm
            arm_connections = [
                (joints['shoulder'], joints['elbow']),
                (joints['elbow'], joints['wrist'])
            ]
            torso_connections = [
                (joints['hip'], joints['shoulder'])
            ]

            joint_positions = {
                'Shoulder': [shoulder[0] * image.shape[1], shoulder[1] * image.shape[0]],
                'Elbow': [elbow[0] * image.shape[1], elbow[1] * image.shape[0]],
                'Wrist': [wrist[0] * image.shape[1], wrist[1] * image.shape[0]],
                'Hip': [hip[0] * image.shape[1], hip[1] * image.shape[0]]
            }

            # Draw arm connections
            for connection in arm_connections:
                start_idx = connection[0].value
                end_idx = connection[1].value

                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]

                start_coords = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
                end_coords = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))

                cv2.line(image, start_coords, end_coords, (0,255,0), 2)

            # Draw torso connections
            for connection in torso_connections:
                start_idx = connection[0].value
                end_idx = connection[1].value

                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]

                start_coords = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
                end_coords = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))

                cv2.line(image, start_coords, end_coords, (0,255,0), 2)

            # Draw joints
            for joint, position in joint_positions.items():
                cv2.circle(image, (int(position[0]), int(position[1])), 7, (0, 0, 255), -1)

            # Display angles with color coding based on correct form
            elbow_color = (255, 255, 255)  # Default white
            shoulder_color = (255, 255, 255)  # Default white

            # Check for form violations
            if shoulder_angle > 150:  # Arm raised too high
                arm_violated[side] = True
                violation_types['arm_angle'] = True
                shoulder_color = (0, 0, 255)  # Red for violation

            if elbow_angle > 170:  # Elbow too straight
                arm_violated[side] = True
                violation_types['elbow'] = True
                elbow_color = (0, 0, 255)  # Red for violation

            # Check arm position (in front of body)
            wrist_x = wrist[0] * image.shape[1]
            shoulder_x = shoulder[0] * image.shape[1]
            wrist_y = wrist[1] * image.shape[0]
            shoulder_y = shoulder[1] * image.shape[0]

            if wrist_y < shoulder_y and abs(wrist_x - shoulder_x) > 100:
                arm_violated[side] = True
                violation_types['arm_position'] = True

            # Display angles
            cv2.putText(
                image,
                f' {int(elbow_angle)}',
                tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                elbow_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                image,
                f' {int(shoulder_angle)}',
                tuple(np.multiply(shoulder, [image.shape[1], image.shape[0]]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                shoulder_color,
                2,
                cv2.LINE_AA
            )

            # Count repetitions when proper form is maintained
            if not arm_violated[side]:
                if side == 'left':
                    if elbow_angle >= 110 and left_state == "down":
                        if wrist_y < shoulder_y and 30 < abs(wrist_x - shoulder_x) < 100:
                            left_state = "up"
                            left_counter += 1
                    elif elbow_angle > 160 and wrist_y > shoulder_y and left_state == "up":
                        left_state = "down"
                elif side == 'right':
                    if elbow_angle >= 110 and right_state == "down":
                        if wrist_y < shoulder_y and 30 < abs(wrist_x - shoulder_x) < 100:
                            right_state = "up"
                            right_counter += 1
                    elif elbow_angle > 160 and wrist_y > shoulder_y and right_state == "up":
                        right_state = "down"

        # Handle audio feedback based on form violations
        current_time = time.time()
        
        # Determine if we should play audio feedback
        new_audio_key = None
        if any(arm_violated.values()):
            feedback_message = ""
            
            # Check for arm angle violation
            if violation_types['arm_angle']:
                if arm_violated['left'] and arm_violated['right']:
                    new_audio_key = "both_shoulders_high"
                    feedback_message = "Lower both arms"
                elif arm_violated['left']:
                    new_audio_key = "left_shoulder_high"
                    feedback_message = "Lower your left arm"
                elif arm_violated['right']:
                    new_audio_key = "right_shoulder_high"
                    feedback_message = "Lower your right arm"
            
            # Check for elbow violation
            elif violation_types['elbow']:
                if arm_violated['left'] and arm_violated['right']:
                    new_audio_key = "both_elbows_straight"
                    feedback_message = "Bend both elbows more"
                elif arm_violated['left']:
                    new_audio_key = "left_elbow_straight"
                    feedback_message = "Bend your left elbow more"
                elif arm_violated['right']:
                    new_audio_key = "right_elbow_straight"
                    feedback_message = "Bend your right elbow more"
            
            # Check for arm position violation
            elif violation_types['arm_position']:
                if arm_violated['left'] and arm_violated['right']:
                    new_audio_key = "both_arms_forward"
                    feedback_message = "Keep both arms in front of your body"
                elif arm_violated['left']:
                    new_audio_key = "left_arm_forward"
                    feedback_message = "Keep your left arm in front of your body"
                elif arm_violated['right']:
                    new_audio_key = "right_arm_forward"
                    feedback_message = "Keep your right arm in front of your body"

        # Update feedback if needed
        if new_audio_key and (current_time - last_feedback_time) > feedback_cooldown:
            current_audio_key = new_audio_key
            current_feedback = feedback_message
            last_feedback_time = current_time

        # Display counters and form status
        cv2.putText(image, f'Left Counter: {left_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Right Counter: {right_counter}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Add form status indicator
        form_status = "GOOD FORM" if not any(arm_violated.values()) else "FIX YOUR FORM"
        form_color = (0, 255, 0) if not any(arm_violated.values()) else (0, 0, 255)
        
        cv2.putText(image, form_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, form_color, 2, cv2.LINE_AA)
        
        # Add exercise guidance
        cv2.putText(image, "Front Raise: Lift arms to shoulder height", (10, image.shape[0] - 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "Keep elbows slightly bent", (10, image.shape[0] - 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "Maximum shoulder angle: 150 degrees", (10, image.shape[0] - 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Return the processed frame and results
    return image, {
        'left_counter': left_counter,
        'right_counter': right_counter,
        'left_state': left_state,
        'right_state': right_state,
        'current_audio_key': current_audio_key,
        'current_feedback': current_feedback,
        'last_feedback_time': last_feedback_time
    } 