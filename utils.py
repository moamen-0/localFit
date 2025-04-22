import numpy as np
import math
import mediapipe as mp
import cv2

# Initialize mediapipe pose with ultra-optimized settings
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create pose instance with highly optimized settings for speed
pose = mp_pose.Pose(
    static_image_mode=False,         # Faster tracking mode
    model_complexity=0,              # Use lightweight model (0)
    smooth_landmarks=True,           # Keep smoothing for quality
    enable_segmentation=False,       # Disable segmentation (not needed)
    min_detection_confidence=0.5,    # Balance between speed and accuracy
    min_tracking_confidence=0.2      # Lower confidence for faster tracking
)

# Create a minimal drawing specification
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1,
    circle_radius=1,
    color=(0, 255, 0)
)

def calculate_angle(a, b, c):
    """
    Ultra-optimized angle calculation between three points
    
    Args:
        a: First point [x, y]
        b: Mid point [x, y]
        c: End point [x, y]
        
    Returns:
        Angle in degrees
    """
    # Avoid numpy array creation if possible
    ab_x = a[0] - b[0]
    ab_y = a[1] - b[1]
    cb_x = c[0] - b[0]
    cb_y = c[1] - b[1]
    
    # Dot product
    dot = ab_x * cb_x + ab_y * cb_y
    
    # Magnitudes
    ab_mag = math.sqrt(ab_x * ab_x + ab_y * ab_y)
    cb_mag = math.sqrt(cb_x * cb_x + cb_y * cb_y)
    
    # Avoid division by zero
    mag_product = ab_mag * cb_mag
    if mag_product < 1e-10:
        return 0
    
    # Cosine value (clipped for stability)
    cos_angle = max(-1.0, min(1.0, dot / mag_product))
    
    # Return angle in degrees
    return math.degrees(math.acos(cos_angle))

def draw_landmarks_lite(image, results):
    """
    Extremely lightweight landmark visualization
    Only draws the minimum necessary landmarks
    
    Args:
        image: Image to draw on
        results: MediaPipe pose results
        
    Returns:
        Image with minimal landmark visualization
    """
    if not results.pose_landmarks:
        return image
    
    h, w, _ = image.shape
    landmarks = results.pose_landmarks.landmark
    
    # Only draw the most essential body parts for exercise monitoring
    essential_connections = [
        # Arms
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        
        # Shoulders
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        
        # Torso
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    ]
    
    # Draw only essential connections with minimal style
    for connection in essential_connections:
        start_idx = connection[0].value
        end_idx = connection[1].value
        
        if 0 <= start_idx < len(landmarks) and 0 <= end_idx < len(landmarks):
            # Draw line with minimal thickness for performance
            cv2.line(
                image, 
                (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)),
                (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)),
                (0, 255, 0), 
                1
            )
    
    # Draw only essential landmarks with minimal radius
    essential_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
    ]
    
    for landmark in essential_landmarks:
        idx = landmark.value
        if 0 <= idx < len(landmarks):
            cv2.circle(
                image,
                (int(landmarks[idx].x * w), int(landmarks[idx].y * h)),
                2,  # Smallest visible radius
                (0, 0, 255),
                -1
            )
    
    return image