import cv2
import numpy as np
import time
import os
import pygame
from utils import calculate_angle, mp_pose, pose

def process_frame(frame, left_counter, right_counter, left_state, right_state, 
                 current_audio_key, current_feedback, last_feedback_time, sound_objects):
    """
    Process a single frame for bicep curl exercise (hammer curl)
    
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
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(frame, (320, 240))  # Lower resolution

    # Variables for tracking violations
    arm_violated = {'left': False, 'right': False}
    violation_types = {'sagittal': False, 'shoulder': False, 'elbow': False}
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

                cv2.line(image, start_coords, end_coords, (0,255,0), 2)  # Different color for torso

            # Draw joints
            for joint, position in joint_positions.items():
                cv2.circle(image, (int(position[0]), int(position[1])), 7, (0, 0, 255), -1)

            # Display angles with color coding based on correct form
            elbow_color = (255, 255, 255)  # Default white
            shoulder_color = (255, 255, 255)  # Default white
            
            # Check if the angles are outside the desired range
            elbow_max = 180
            shoulder_max = 30
            sagittal_angle_threshold = 90
            shoulder_max_back = 25  # Maximum angle for shoulder extension backward
            elbow_min_back = 0  

            # Change color if angle is in violation
            if elbow_angle > elbow_max:
                elbow_color = (0, 0, 255)  # Red for violation
            
            if shoulder_angle > shoulder_max or shoulder_angle > sagittal_angle_threshold:
                shoulder_color = (0, 0, 255)  # Red for violation
            
            # Check for specific violations and track them
            if elbow_angle > elbow_max:
                arm_violated[side] = True
                violation_types['elbow'] = True
            if shoulder_angle >= shoulder_max:
                arm_violated[side] = True
                violation_types['shoulder'] = True
            if elbow_angle < elbow_min_back or shoulder_angle > shoulder_max_back:
                arm_violated[side] = True
            if shoulder_angle > sagittal_angle_threshold:
                arm_violated[side] = True
                violation_types['sagittal'] = True
            
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
            if not arm_violated['left'] and not arm_violated['right']:
                if side == 'left':
                    if elbow_angle > 160:
                        left_state = 'down'
                    if elbow_angle < 30 and left_state == 'down':
                        left_state = 'up'
                        left_counter += 1
                        print(f'Left Counter: {left_counter}')
                if side == 'right':
                    if elbow_angle > 160:
                        right_state = 'down'
                    if elbow_angle < 30 and right_state == 'down':
                        right_state = 'up'
                        right_counter += 1
                        print(f'Right Counter: {right_counter}')

        # Handle audio feedback based on form violations
        current_time = time.time()
        
        # Determine if we should play audio feedback
        new_audio_key = None
        if any(arm_violated.values()):
            feedback_message = ""
            
            # Check for sagittal angle violation (arm forward)
            if violation_types['sagittal']:
                if arm_violated['left'] and arm_violated['right']:
                    new_audio_key = "both_arms_forward"
                    feedback_message = "Keep both arms closer to your body"
                elif arm_violated['left']:
                    new_audio_key = "left_arm_forward"
                    feedback_message = "Keep your left arm closer to your body"
                elif arm_violated['right']:
                    new_audio_key = "right_arm_forward"
                    feedback_message = "Keep your right arm closer to your body"
            
            # Check for shoulder high violation
            elif violation_types['shoulder']:
                if arm_violated['left'] and arm_violated['right']:
                    new_audio_key = "both_shoulders_high"
                    feedback_message = "Lower both shoulders"
                elif arm_violated['left']:
                    new_audio_key = "left_shoulder_high"
                    feedback_message = "Lower your left shoulder"
                elif arm_violated['right']:
                    new_audio_key = "right_shoulder_high"
                    feedback_message = "Lower your right shoulder"
            
            # Check for elbow straight violation
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
            
            # If we have a message to play and it's either a new message or enough time has passed
            if new_audio_key and (new_audio_key != current_audio_key or current_time - last_feedback_time > feedback_cooldown):
                # For server we just update state, the client is responsible for playing the sound
                current_audio_key = new_audio_key
                current_feedback = feedback_message
                last_feedback_time = current_time
        else:
            # If form is correct, clear any audio feedback
            if current_audio_key is not None:
                current_audio_key = None
                current_feedback = ""

        # Draw counters on the image
        cv2.putText(image, f'Left Counter: {left_counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Right Counter: {right_counter}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display current feedback message if active
        if current_feedback:
            # Add centered text with background for better visibility
            text_size = cv2.getTextSize(current_feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = image.shape[0] - 50  # Position at bottom of screen
            
            # Draw semi-transparent background for text
            overlay = image.copy()
            cv2.rectangle(overlay, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Draw text
            cv2.putText(image, current_feedback, (text_x, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Return the processed frame and updated state
    return image, {
        'left_counter': left_counter,
        'right_counter': right_counter,
        'left_state': left_state,
        'right_state': right_state,
        'current_audio_key': current_audio_key,
        'current_feedback': current_feedback,
        'last_feedback_time': last_feedback_time
    }

# Keep the original function for backwards compatibility
def hummer(sound):
    """
    Track bicep curl exercise (hammer curl)
    
    Args:
        sound: Pygame sound object for alerts
        
    Yields:
        Video frames with pose tracking
    """
    # Initialize pygame for sound if not already done
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    
    # Variables for tracking
    left_counter = 0  # Counter for left arm
    right_counter = 0  # Counter for right arm
    left_state = None  # State for left arm
    right_state = None  # State for right arm
    current_audio_key = None
    current_feedback = ""
    last_feedback_time = 0
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        # Pre-defined audio feedback messages (minimal for compatibility)
        audio_messages = {
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
        
        # Process the frame using our new function
        # Create dummy sound objects for compatibility
        sound_objects = {}
        for key in audio_messages.keys():
            sound_objects[key] = sound  # Use the same sound for all keys
            
        processed_frame, results = process_frame(
            frame, 
            left_counter, 
            right_counter,
            left_state,
            right_state,
            current_audio_key,
            current_feedback,
            last_feedback_time,
            sound_objects
        )
        
        # Update state
        left_counter = results['left_counter']
        right_counter = results['right_counter']
        left_state = results['left_state']
        right_state = results['right_state']
        current_audio_key = results['current_audio_key']
        current_feedback = results['current_feedback']
        last_feedback_time = results['last_feedback_time']
        
        # Play sound if needed (in the original function)
        if current_audio_key and sound:
            sound.play()
        
        # Convert the image to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        # Yield the frame to the Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
               
    # Clean up
    cap.release()
    pygame.mixer.stop()