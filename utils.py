import numpy as np
import math
import mediapipe as mp
import cv2

# Initialize mediapipe pose with optimized settings
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create pose instance with optimized settings
# Increasing min_detection_confidence helps reduce jitter
# Using static_image_mode=False for better tracking performance in video streams
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,  # Use medium complexity model for balance of performance and accuracy
    smooth_landmarks=True,  # Enable landmark smoothing to reduce jitter
    enable_segmentation=False,  # Disable segmentation for better performance
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Create a more efficient landmark drawing specification
# This reduces the visual elements for better performance
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,
    circle_radius=1,
    color=(0, 255, 0)
)

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points - optimized version
    
    Args:
        a: First point [x, y]
        b: Mid point [x, y]
        c: End point [x, y]
        
    Returns:
        Angle in degrees
    """
    # Convert to numpy arrays if they aren't already
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    # Calculate vectors
    ab = a - b
    bc = c - b
    
    # Use dot product formula to get the angle
    dot_product = np.dot(ab, bc)
    norm_product = np.linalg.norm(ab) * np.linalg.norm(bc)
    
    # Ensure the argument to arccos is within valid range [-1, 1]
    # This prevents numerical instabilities
    cosine = np.clip(dot_product / (norm_product + 1e-10), -1.0, 1.0)
    
    # Convert to degrees
    angle = math.degrees(np.arccos(cosine))
    
    return angle

def draw_landmarks_lite(image, results):
    """
    Draw landmarks with minimal overhead for better performance
    
    Args:
        image: Image to draw on
        results: MediaPipe pose results
        
    Returns:
        Image with landmarks drawn
    """
    if results.pose_landmarks:
        # Only draw important landmarks for visualization
        important_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        # Draw only key connections
        key_connections = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
        ]
        
        h, w, _ = image.shape
        landmarks = results.pose_landmarks.landmark
        
        # Draw key connections
        for connection in key_connections:
            start_idx = connection[0].value
            end_idx = connection[1].value
            
            if 0 <= start_idx < len(landmarks) and 0 <= end_idx < len(landmarks):
                cv2.line(
                    image, 
                    (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)),
                    (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)),
                    (0, 255, 0), 
                    2
                )
        
        # Draw important landmarks
        for landmark in important_landmarks:
            idx = landmark.value
            if 0 <= idx < len(landmarks):
                cv2.circle(
                    image,
                    (int(landmarks[idx].x * w), int(landmarks[idx].y * h)),
                    5,
                    (0, 0, 255),
                    -1
                )
    
    return image