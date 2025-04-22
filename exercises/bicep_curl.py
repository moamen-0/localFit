import cv2
import numpy as np
import time
import os
import pygame
from utils import calculate_angle, mp_pose, pose

# Optimization flags - tune these based on your needs
ENABLE_FULL_RENDERING = False  # Set to False for maximum performance
ENABLE_ANGLE_DISPLAY = False   # Set to False to skip drawing angles
PROCESS_EVERY_N_FRAMES = 2     # Process every Nth frame fully
DISPLAY_MINIMAL_FEEDBACK = True  # Only show essential feedback

def process_frame(frame, left_counter, right_counter, left_state, right_state, 
                 current_audio_key, current_feedback, last_feedback_time, sound_objects):
    """
    Process a single frame for bicep curl exercise with extreme optimization
    
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
    # Check if frame is None or empty
    if frame is None or frame.size == 0:
        # Return empty results
        return frame, {
            'left_counter': left_counter,
            'right_counter': right_counter,
            'left_state': left_state,
            'right_state': right_state,
            'current_audio_key': current_audio_key,
            'current_feedback': current_feedback,
            'last_feedback_time': last_feedback_time
        }
    
    # Resize for faster processing - aggressively reduce size
    height, width = frame.shape[:2]
    max_dim = 320  # Maximum dimension (very small for speed)
    
    if width > max_dim:
        scale_factor = max_dim / width
        frame = cv2.resize(frame, (max_dim, int(height * scale_factor)), 
                           interpolation=cv2.INTER_NEAREST)  # Fastest interpolation
    
    # Convert to RGB (required by MediaPipe)
    # Use direct conversion without copying
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set image as not writeable to improve performance
    image.flags.writeable = False
    
    # Process with MediaPipe
    results = pose.process(image)
    
    # Convert back to BGR for OpenCV display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Variables for tracking violations
    arm_violated = {'left': False, 'right': False}
    violation_types = {'sagittal': False, 'shoulder': False, 'elbow': False}
    feedback_cooldown = 3  # Extend cooldown to reduce frequent feedback
    
    # If no landmarks detected, return original frame with counters
    if not results.pose_landmarks:
        # Draw minimal information (just counters)
        cv2.putText(image, f'L: {left_counter} R: {right_counter}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return image, {
            'left_counter': left_counter,
            'right_counter': right_counter,
            'left_state': left_state,
            'right_state': right_state,
            'current_audio_key': current_audio_key,
            'current_feedback': current_feedback,
            'last_feedback_time': last_feedback_time
        }
    
    # Process pose landmarks
    landmarks = results.pose_landmarks.landmark
    
    # Define arm landmarks
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
    
    # Extract coordinates - minimal processing for core functionality
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    
    # Minimal drawing (only if full rendering enabled)
    if ENABLE_FULL_RENDERING:
        # Draw minimal connections
        h, w = image.shape[:2]
        
        # Draw a line between shoulders
        cv2.line(
            image,
            (int(left_shoulder[0] * w), int(left_shoulder[1] * h)),
            (int(right_shoulder[0] * w), int(right_shoulder[1] * h)),
            (0, 255, 255), 1
        )
        
        # Draw a line between hips
        cv2.line(
            image,
            (int(left_hip[0] * w), int(left_hip[1] * h)),
            (int(right_hip[0] * w), int(right_hip[1] * h)),
            (0, 255, 255), 1
        )
    
    # Process each arm (left and right)
    for side, joints in arm_sides.items():
        # Get coordinates
        shoulder = [landmarks[joints['shoulder'].value].x, landmarks[joints['shoulder'].value].y]
        elbow = [landmarks[joints['elbow'].value].x, landmarks[joints['elbow'].value].y]
        wrist = [landmarks[joints['wrist'].value].x, landmarks[joints['wrist'].value].y]
        hip = [landmarks[joints['hip'].value].x, landmarks[joints['hip'].value].y]
        
        # Calculate angles - core functionality
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = calculate_angle(hip, shoulder, elbow)
        
        # Only draw if full rendering enabled
        if ENABLE_FULL_RENDERING:
            h, w = image.shape[:2]
            
            # Draw minimal arm connections
            cv2.line(image, (int(shoulder[0] * w), int(shoulder[1] * h)), 
                     (int(elbow[0] * w), int(elbow[1] * h)), (0, 255, 0), 1)
            cv2.line(image, (int(elbow[0] * w), int(elbow[1] * h)), 
                     (int(wrist[0] * w), int(wrist[1] * h)), (0, 255, 0), 1)
            
            # Draw joint circles (smaller)
            cv2.circle(image, (int(shoulder[0] * w), int(shoulder[1] * h)), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(elbow[0] * w), int(elbow[1] * h)), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(wrist[0] * w), int(wrist[1] * h)), 3, (0, 0, 255), -1)
            
            # Display angles if enabled
            if ENABLE_ANGLE_DISPLAY:
                # Text size reduced
                cv2.putText(image, f'{int(elbow_angle)}', 
                          (int(elbow[0] * w), int(elbow[1] * h)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(image, f'{int(shoulder_angle)}', 
                          (int(shoulder[0] * w), int(shoulder[1] * h)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Define thresholds - core functionality
        elbow_max = 180
        shoulder_max = 30
        sagittal_angle_threshold = 90
        shoulder_max_back = 25
        elbow_min_back = 0
        
        # Check for violations - core functionality
        if elbow_angle > elbow_max:
            arm_violated[side] = True
            violation_types['elbow'] = True
        
        if shoulder_angle > shoulder_max or shoulder_angle > sagittal_angle_threshold:
            arm_violated[side] = True
            violation_types['shoulder'] = True
            if shoulder_angle > sagittal_angle_threshold:
                violation_types['sagittal'] = True
        
        if elbow_angle < elbow_min_back or shoulder_angle > shoulder_max_back:
            arm_violated[side] = True
        
        # Rep counting logic - core functionality
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
    
    # Feedback logic - core functionality
    current_time = time.time()
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
        
        # Only update feedback if cooldown has elapsed
        if new_audio_key and (new_audio_key != current_audio_key or current_time - last_feedback_time > feedback_cooldown):
            current_audio_key = new_audio_key
            current_feedback = feedback_message
            last_feedback_time = current_time
    else:
        # Clear feedback if form is correct
        if current_audio_key:
            current_audio_key = None
            current_feedback = ""
    
    # Draw counters - always show this minimal info
    cv2.putText(image, f'L: {left_counter} R: {right_counter}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Display feedback if enabled
    if DISPLAY_MINIMAL_FEEDBACK and current_feedback:
        # Simplified feedback display
        cv2.putText(image, current_feedback, 
                  (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Return processed frame and updated state
    return image, {
        'left_counter': left_counter,
        'right_counter': right_counter,
        'left_state': left_state,
        'right_state': right_state,
        'current_audio_key': current_audio_key,
        'current_feedback': current_feedback,
        'last_feedback_time': last_feedback_time
    }

# Backward compatibility function
def hummer(sound):
    """
    Simplified version of hummer function for backward compatibility
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
    
    # Create dummy sound objects
    sound_objects = {}
    
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Process every Nth frame fully to save CPU
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            # Just draw counters on non-processed frames
            cv2.putText(frame, f'L: {left_counter} R: {right_counter}', 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                      
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            jpeg_frame = buffer.tobytes()
            
            # Yield the frame
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')
            continue
        
        # Process the frame
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
        
        # Play sound if needed
        if current_audio_key and sound:
            sound.play()
        
        # Convert to JPEG with low quality for performance
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        jpeg_frame = buffer.tobytes()
        
        # Yield the frame to the Flask response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')
    
    # Clean up
    cap.release()
    pygame.mixer.stop()