import numpy as np
import math
import mediapipe as mp

# Initialize mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create pose instance
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    
    Args:
        a: First point [x, y]
        b: Mid point [x, y]
        c: End point [x, y]
        
    Returns:
        Angle in degrees
    """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    ab = a - b
    bc = c - b

    # Calculate the angle
    angle = np.arccos(np.clip(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)), -1.0, 1.0))
    return math.degrees(angle)