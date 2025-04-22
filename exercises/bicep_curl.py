import cv2
import numpy as np
import time
import os
import pygame
from utils import calculate_angle, mp_pose, pose

# Global settings for MediaPipe optimization
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

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
    # Resize for faster processing if needed
    height, width = frame.shape[:2]
    if width > 640:
        frame = cv2.resize(frame, (640, int(height * 640 / width)))
    
    # Process the image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set static image flag for better performance
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    
    # Convert back to BGR and resize for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
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

        # Process each arm
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

            # Draw arm connections - simplified for performance
            # Just draw key lines instead of all connections
            cv2.line(
                image,
                (int(shoulder[0] * image.shape[1]), int(shoulder[1] * image.shape[0])),
                (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0])),
                (0, 255, 0), 
                2
            )
            cv2.line(
                image,
                (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0])),
                (int(wrist[0] * image.shape[1]), int(wrist[1] * image.shape[0])),
                (0, 255, 0), 
                2
            )
            cv2.line(
                image,
                (int(shoulder[0] * image.shape[1]), int(shoulder[1] * image.shape[0])),
                (int(hip[0] * image.shape[1]), int(hip[1] * image.shape[0])),
                (0, 255, 0),
                2
            )
            
            # Draw only important joints
            cv2.circle(
                image, 
                (int(shoulder[0] * image.shape[1]), int(shoulder[1] * image.shape[0])), 
                5, (0, 0, 255), -1
            )
            cv2.circle(
                image, 
                (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0])), 
                5, (0, 0, 255), -1
            )
            cv2.circle(
                image, 
                (int(wrist[0] * image.shape[1]), int(wrist[1] * image.shape[0])), 
                5, (0, 0, 255), -1
            )

            # Check form and angles
            # Define angle thresholds
            elbow_max = 180
            shoulder_max = 30
            sagittal_angle_threshold = 90
            shoulder_max_back = 25  # Maximum angle for shoulder extension backward
            elbow_min_back = 0

            # Display only critical angles with basic formatting
            # Use a simple text rather than fancy formatting for better performance
            elbow_color = (255, 255, 255)  # Default white
            shoulder_color = (255, 255, 255)  # Default white
            
            # Change color if angle is in violation
            if elbow_angle > elbow_max:
                elbow_color = (0, 0, 255)  # Red for violation
                arm_violated[side] = True
                violation_types['elbow'] = True
            
            if shoulder_angle > shoulder_max or shoulder_angle > sagittal_angle_threshold:
                shoulder_color = (0, 0, 255)  # Red for violation
                arm_violated[side] = True
                violation_types['shoulder'] = True
                if shoulder_angle > sagittal_angle_threshold:
                    violation_types['sagittal'] = True
            
            if elbow_angle < elbow_min_back or shoulder_angle > shoulder_max_back:
                arm_violated[side] = True
            
            # Simplified text display
            cv2.putText(
                image,
                f'{int(elbow_angle)}',
                (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                elbow_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                image,
                f'{int(shoulder_angle)}',
                (int(shoulder[0] * image.shape[1]), int(shoulder[1] * image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                shoulder_color,
                2,
                cv2.LINE_AA
            )

            # Rep counting logic
            if not arm_violated[side]:
                if side == 'left':
                    if elbow_angle > 160:
                        left_state = 'down'
                    if elbow_angle < 30 and left_state == 'down':
                        left_state = 'up'
                        left_counter += 1
                elif side == 'right':
                    if elbow_angle > 160:
                        right_state = 'down'
                    if elbow_angle < 30 and right_state == 'down':
                        right_state = 'up'
                        right_counter += 1

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
            
            # If we have a message to play and enough time has passed
            if new_audio_key and (new_audio_key != current_audio_key or current_time - last_feedback_time > feedback_cooldown):
                current_audio_key = new_audio_key
                current_feedback = feedback_message
                last_feedback_time = current_time
        else:
            # If form is correct, clear any audio feedback
            if current_audio_key is not None:
                current_audio_key = None
                current_feedback = ""
                
        # Draw counters on the image - simplified text display
        cv2.putText(
            image, 
            f'L: {left_counter}  R: {right_counter}', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 0, 0), 
            2, 
            cv2.LINE_AA
        )
        
        # Display feedback in a more efficient way
        if current_feedback:
            # Simplified feedback display
            text_size = cv2.getTextSize(current_feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = image.shape[0] - 20
            
            # Black background rectangle for text visibility
            cv2.rectangle(
                image,
                (text_x - 5, text_y - 20),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                image,
                current_feedback,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

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

# Keep the original function for backwards compatibility but optimize it
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
    left_counter = 0
    right_counter = 0
    left_state = None
    right_state = None
    current_audio_key = None
    current_feedback = ""
    last_feedback_time = 0
    
    cap = cv2.VideoCapture(0)
    
    # Optimize camera settings if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create dummy sound objects for compatibility
    sound_objects = {}
    for key in ["left_arm_forward", "right_arm_forward", "both_arms_forward", 
                "left_shoulder_high", "right_shoulder_high", "both_shoulders_high",
                "left_elbow_straight", "right_elbow_straight", "both_elbows_straight"]:
        sound_objects[key] = sound
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Process the frame using our optimized function
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
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame = buffer.tobytes()

        # Yield the frame to the Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
               
    # Clean up
    cap.release()
    pygame.mixer.stop()