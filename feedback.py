

import numpy as np
import math

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def check_barbell_biceps_curl(keypoints_with_scores):
    """Check if the form for a barbell biceps curl exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Extract keypoint coordinates
    kpts_x = keypoints_with_scores[0, 0, :, 1]
    kpts_y = keypoints_with_scores[0, 0, :, 0]
    kpts_scores = keypoints_with_scores[0, 0, :, 2]

    # Check alignment of elbows and wrists
    left_elbow = (kpts_x[KEYPOINT_DICT['left_elbow']], kpts_y[KEYPOINT_DICT['left_elbow']])
    right_elbow = (kpts_x[KEYPOINT_DICT['right_elbow']], kpts_y[KEYPOINT_DICT['right_elbow']])
    left_wrist = (kpts_x[KEYPOINT_DICT['left_wrist']], kpts_y[KEYPOINT_DICT['left_wrist']])
    right_wrist = (kpts_x[KEYPOINT_DICT['right_wrist']], kpts_y[KEYPOINT_DICT['right_wrist']])
    left_elbow_angle = calculate_angle(left_elbow, left_wrist, (left_wrist[0], left_elbow[1]))
    right_elbow_angle = calculate_angle(right_elbow, right_wrist, (right_wrist[0], right_elbow[1]))

    if left_elbow_angle < 160 or right_elbow_angle < 160:
        feedback.append("Elbows are not properly aligned with the wrists.")
    else:
        feedback.append("Elbows are properly aligned with the wrists.")

    return feedback


def check_bench_press(keypoints_with_scores):
    """Check if the form for a bench press exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Extract keypoint coordinates
    kpts_x = keypoints_with_scores[0, 0, :, 1]
    kpts_y = keypoints_with_scores[0, 0, :, 0]
    kpts_scores = keypoints_with_scores[0, 0, :, 2]

    # Check alignment of elbows and shoulders
    left_elbow = (kpts_x[KEYPOINT_DICT['left_elbow']], kpts_y[KEYPOINT_DICT['left_elbow']])
    right_elbow = (kpts_x[KEYPOINT_DICT['right_elbow']], kpts_y[KEYPOINT_DICT['right_elbow']])
    left_shoulder = (kpts_x[KEYPOINT_DICT['left_shoulder']], kpts_y[KEYPOINT_DICT['left_shoulder']])
    right_shoulder = (kpts_x[KEYPOINT_DICT['right_shoulder']], kpts_y[KEYPOINT_DICT['right_shoulder']])
    left_elbow_shoulder_angle = calculate_angle(left_elbow, left_shoulder, (left_shoulder[0], left_elbow[1]))
    right_elbow_shoulder_angle = calculate_angle(right_elbow, right_shoulder, (right_shoulder[0], right_elbow[1]))

    if left_elbow_shoulder_angle < 150 or right_elbow_shoulder_angle < 150:
        feedback.append("Elbows are not properly aligned with the shoulders.")
    else:
        feedback.append("Elbows are properly aligned with the shoulders.")

    # Check if wrists are in line with shoulders
    left_wrist = (kpts_x[KEYPOINT_DICT['left_wrist']], kpts_y[KEYPOINT_DICT['left_wrist']])
    right_wrist = (kpts_x[KEYPOINT_DICT['right_wrist']], kpts_y[KEYPOINT_DICT['right_wrist']])
    left_shoulder_wrist_angle = calculate_angle(left_wrist, left_shoulder, (left_shoulder[0], left_wrist[1]))
    right_shoulder_wrist_angle = calculate_angle(right_wrist, right_shoulder, (right_shoulder[0], right_wrist[1]))

    if left_shoulder_wrist_angle < 150 or right_shoulder_wrist_angle < 150:
        feedback.append("Wrists are not properly aligned with the shoulders.")
    else:
        feedback.append("Wrists are properly aligned with the shoulders.")

    return feedback

def check_chest_fly_machine(keypoints_with_scores):
    """Check if the form for a chest fly machine exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Extract keypoint coordinates
    kpts_x = keypoints_with_scores[0, 0, :, 1]
    kpts_y = keypoints_with_scores[0, 0, :, 0]
    kpts_scores = keypoints_with_scores[0, 0, :, 2]

    # Check alignment of elbows and shoulders
    left_elbow = (kpts_x[KEYPOINT_DICT['left_elbow']], kpts_y[KEYPOINT_DICT['left_elbow']])
    right_elbow = (kpts_x[KEYPOINT_DICT['right_elbow']], kpts_y[KEYPOINT_DICT['right_elbow']])
    left_shoulder = (kpts_x[KEYPOINT_DICT['left_shoulder']], kpts_y[KEYPOINT_DICT['left_shoulder']])
    right_shoulder = (kpts_x[KEYPOINT_DICT['right_shoulder']], kpts_y[KEYPOINT_DICT['right_shoulder']])
    left_elbow_shoulder_angle = calculate_angle(left_elbow, left_shoulder, (left_shoulder[0], left_elbow[1]))
    right_elbow_shoulder_angle = calculate_angle(right_elbow, right_shoulder, (right_shoulder[0], right_elbow[1]))

    if left_elbow_shoulder_angle < 150 or right_elbow_shoulder_angle < 150:
        feedback.append("Elbows are not properly aligned with the shoulders.")
    else:
        feedback.append("Elbows are properly aligned with the shoulders.")

    # Check if wrists are in line with shoulders
    left_wrist = (kpts_x[KEYPOINT_DICT['left_wrist']], kpts_y[KEYPOINT_DICT['left_wrist']])
    right_wrist = (kpts_x[KEYPOINT_DICT['right_wrist']], kpts_y[KEYPOINT_DICT['right_wrist']])
    left_shoulder_wrist_angle = calculate_angle(left_wrist, left_shoulder, (left_shoulder[0], left_wrist[1]))
    right_shoulder_wrist_angle = calculate_angle(right_wrist, right_shoulder, (right_shoulder[0], right_wrist[1]))

    if left_shoulder_wrist_angle < 150 or right_shoulder_wrist_angle < 150:
        feedback.append("Wrists are not properly aligned with the shoulders.")
    else:
        feedback.append("Wrists are properly aligned with the shoulders.")

    return feedback

def check_chest_fly_machine(keypoints_with_scores):
    """Check if the form for a chest fly machine exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Extract keypoint coordinates
    kpts_x = keypoints_with_scores[0, 0, :, 1]
    kpts_y = keypoints_with_scores[0, 0, :, 0]
    kpts_scores = keypoints_with_scores[0, 0, :, 2]

    # Check alignment of elbows and shoulders
    left_elbow = (kpts_x[KEYPOINT_DICT['left_elbow']], kpts_y[KEYPOINT_DICT['left_elbow']])
    right_elbow = (kpts_x[KEYPOINT_DICT['right_elbow']], kpts_y[KEYPOINT_DICT['right_elbow']])
    left_shoulder = (kpts_x[KEYPOINT_DICT['left_shoulder']], kpts_y[KEYPOINT_DICT['left_shoulder']])
    right_shoulder = (kpts_x[KEYPOINT_DICT['right_shoulder']], kpts_y[KEYPOINT_DICT['right_shoulder']])
    left_elbow_shoulder_angle = calculate_angle(left_elbow, left_shoulder, (left_shoulder[0], left_elbow[1]))
    right_elbow_shoulder_angle = calculate_angle(right_elbow, right_shoulder, (right_shoulder[0], right_elbow[1]))

    if left_elbow_shoulder_angle < 150 or right_elbow_shoulder_angle < 150:
        feedback.append("Elbows are not properly aligned with the shoulders.")
    else:
        feedback.append("Elbows are properly aligned with the shoulders.")

    # Check if wrists are in line with shoulders
    left_wrist = (kpts_x[KEYPOINT_DICT['left_wrist']], kpts_y[KEYPOINT_DICT['left_wrist']])
    right_wrist = (kpts_x[KEYPOINT_DICT['right_wrist']], kpts_y[KEYPOINT_DICT['right_wrist']])
    left_shoulder_wrist_angle = calculate_angle(left_wrist, left_shoulder, (left_shoulder[0], left_wrist[1]))
    right_shoulder_wrist_angle = calculate_angle(right_wrist, right_shoulder, (right_shoulder[0], right_wrist[1]))

    if left_shoulder_wrist_angle < 150 or right_shoulder_wrist_angle < 150:
        feedback.append("Wrists are not properly aligned with the shoulders.")
    else:
        feedback.append("Wrists are properly aligned with the shoulders.")

    return feedback



def check_decline_bench_press(keypoints_with_scores):
    """Check if the form for a decline bench press exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Extract keypoint coordinates
    kpts_x = keypoints_with_scores[0, 0, :, 1]
    kpts_y = keypoints_with_scores[0, 0, :, 0]
    kpts_scores = keypoints_with_scores[0, 0, :, 2]

    # Check alignment of shoulders, elbows, and wrists
    left_shoulder = (kpts_x[KEYPOINT_DICT['left_shoulder']], kpts_y[KEYPOINT_DICT['left_shoulder']])
    right_shoulder = (kpts_x[KEYPOINT_DICT['right_shoulder']], kpts_y[KEYPOINT_DICT['right_shoulder']])
    left_elbow = (kpts_x[KEYPOINT_DICT['left_elbow']], kpts_y[KEYPOINT_DICT['left_elbow']])
    right_elbow = (kpts_x[KEYPOINT_DICT['right_elbow']], kpts_y[KEYPOINT_DICT['right_elbow']])
    left_wrist = (kpts_x[KEYPOINT_DICT['left_wrist']], kpts_y[KEYPOINT_DICT['left_wrist']])
    right_wrist = (kpts_x[KEYPOINT_DICT['right_wrist']], kpts_y[KEYPOINT_DICT['right_wrist']])
    
    left_shoulder_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_shoulder_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    if left_shoulder_elbow_angle < 150 or right_shoulder_elbow_angle < 150:
        feedback.append("Shoulders and elbows are not properly aligned.")
    else:
        feedback.append("Shoulders and elbows are properly aligned.")

    return feedback

def check_hammer_curl(keypoints_with_scores):
    """Check if the form for a hammer curl exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Extract keypoint coordinates
    kpts_x = keypoints_with_scores[0, 0, :, 1]
    kpts_y = keypoints_with_scores[0, 0, :, 0]
    kpts_scores = keypoints_with_scores[0, 0, :, 2]

    # Check alignment of shoulders, elbows, and wrists
    left_shoulder = (kpts_x[KEYPOINT_DICT['left_shoulder']], kpts_y[KEYPOINT_DICT['left_shoulder']])
    right_shoulder = (kpts_x[KEYPOINT_DICT['right_shoulder']], kpts_y[KEYPOINT_DICT['right_shoulder']])
    left_elbow = (kpts_x[KEYPOINT_DICT['left_elbow']], kpts_y[KEYPOINT_DICT['left_elbow']])
    right_elbow = (kpts_x[KEYPOINT_DICT['right_elbow']], kpts_y[KEYPOINT_DICT['right_elbow']])
    left_wrist = (kpts_x[KEYPOINT_DICT['left_wrist']], kpts_y[KEYPOINT_DICT['left_wrist']])
    right_wrist = (kpts_x[KEYPOINT_DICT['right_wrist']], kpts_y[KEYPOINT_DICT['right_wrist']])
    
    left_shoulder_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_shoulder_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    if left_shoulder_elbow_angle < 150 or right_shoulder_elbow_angle < 150:
        feedback.append("Shoulders and elbows are not properly aligned.")
    else:
        feedback.append("Shoulders and elbows are properly aligned.")

    return feedback


def check_hip_thrust(keypoints_with_scores):
    """Check if the form for a hip thrust exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_hip_index = 11
    right_hip_index = 12
    left_knee_index = 13
    right_knee_index = 14

    # Extract keypoint coordinates
    left_hip = keypoints_with_scores[0, 0, left_hip_index, :2]  # x, y coordinates
    right_hip = keypoints_with_scores[0, 0, right_hip_index, :2]
    left_knee = keypoints_with_scores[0, 0, left_knee_index, :2]
    right_knee = keypoints_with_scores[0, 0, right_knee_index, :2]

    # Check alignment of hips and knees
    hip_alignment_threshold = 0.1  # Adjust as needed
    if abs(left_hip[0] - right_hip[0]) > hip_alignment_threshold:
        feedback.append("Align your hips properly.")

    knee_alignment_threshold = 0.1  # Adjust as needed
    if abs(left_knee[0] - right_knee[0]) > knee_alignment_threshold:
        feedback.append("Align your knees properly.")

    return feedback

def check_incline_bench_press(keypoints_with_scores):
    """Check if the form for an incline bench press exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_elbow_index = 7
    right_elbow_index = 8

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_elbow = keypoints_with_scores[0, 0, left_elbow_index, :2]
    right_elbow = keypoints_with_scores[0, 0, right_elbow_index, :2]

    # Check the angle between arms and torso
    torso_angle_threshold = 150  # Adjust as needed
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, [0, 0])
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, [0, 0])
    if left_arm_angle < torso_angle_threshold or right_arm_angle < torso_angle_threshold:
        feedback.append("Keep your arms at an appropriate angle to your torso.")

    return feedback

def check_lat_pulldown(keypoints_with_scores):
    """Check if the form for a lat pulldown exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_elbow_index = 7
    right_elbow_index = 8
    left_wrist_index = 9
    right_wrist_index = 10

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_elbow = keypoints_with_scores[0, 0, left_elbow_index, :2]
    right_elbow = keypoints_with_scores[0, 0, right_elbow_index, :2]
    left_wrist = keypoints_with_scores[0, 0, left_wrist_index, :2]
    right_wrist = keypoints_with_scores[0, 0, right_wrist_index, :2]

    # Check the alignment of wrists with elbows
    wrist_elbow_alignment_threshold = 50  # Adjust as needed
    left_wrist_elbow_dist = np.linalg.norm(left_wrist - left_elbow)
    right_wrist_elbow_dist = np.linalg.norm(right_wrist - right_elbow)
    if left_wrist_elbow_dist > wrist_elbow_alignment_threshold or right_wrist_elbow_dist > wrist_elbow_alignment_threshold:
        feedback.append("Align your wrists with your elbows properly.")

    # Check the angle between arms and torso
    torso_angle_threshold = 150  # Adjust as needed
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if left_arm_angle < torso_angle_threshold or right_arm_angle < torso_angle_threshold:
        feedback.append("Keep your arms at an appropriate angle to your torso.")

    return feedback

def check_lateral_raises(keypoints_with_scores):
    """Check if the form for lateral raises exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_elbow_index = 7
    right_elbow_index = 8
    left_wrist_index = 9
    right_wrist_index = 10

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_elbow = keypoints_with_scores[0, 0, left_elbow_index, :2]
    right_elbow = keypoints_with_scores[0, 0, right_elbow_index, :2]
    left_wrist = keypoints_with_scores[0, 0, left_wrist_index, :2]
    right_wrist = keypoints_with_scores[0, 0, right_wrist_index, :2]

    # Check the alignment of wrists with elbows
    wrist_elbow_alignment_threshold = 50  # Adjust as needed
    left_wrist_elbow_dist = np.linalg.norm(left_wrist - left_elbow)
    right_wrist_elbow_dist = np.linalg.norm(right_wrist - right_elbow)
    if left_wrist_elbow_dist > wrist_elbow_alignment_threshold or right_wrist_elbow_dist > wrist_elbow_alignment_threshold:
        feedback.append("Align your wrists with your elbows properly.")

    # Check the angle between arms and torso
    torso_angle_threshold = 150  # Adjust as needed
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if left_arm_angle < torso_angle_threshold or right_arm_angle < torso_angle_threshold:
        feedback.append("Keep your arms at an appropriate angle to your torso.")

    return feedback


def check_leg_extension(keypoints_with_scores):
    """Check if the form for leg extension exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_hip_index = 11
    right_hip_index = 12
    left_knee_index = 13
    right_knee_index = 14
    left_ankle_index = 15
    right_ankle_index = 16

    # Extract keypoint coordinates
    left_hip = keypoints_with_scores[0, 0, left_hip_index, :2]  # x, y coordinates
    right_hip = keypoints_with_scores[0, 0, right_hip_index, :2]
    left_knee = keypoints_with_scores[0, 0, left_knee_index, :2]
    right_knee = keypoints_with_scores[0, 0, right_knee_index, :2]
    left_ankle = keypoints_with_scores[0, 0, left_ankle_index, :2]
    right_ankle = keypoints_with_scores[0, 0, right_ankle_index, :2]

    # Check the alignment of knees with hips
    knee_hip_alignment_threshold = 50  # Adjust as needed
    left_knee_hip_dist = np.linalg.norm(left_knee - left_hip)
    right_knee_hip_dist = np.linalg.norm(right_knee - right_hip)
    if left_knee_hip_dist > knee_hip_alignment_threshold or right_knee_hip_dist > knee_hip_alignment_threshold:
        feedback.append("Align your knees with your hips properly.")

    # Check the angle between lower leg and upper leg
    leg_angle_threshold = 120  # Adjust as needed
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if left_leg_angle < leg_angle_threshold or right_leg_angle < leg_angle_threshold:
        feedback.append("Keep your legs at an appropriate angle to your body.")

    return feedback

def check_leg_raises(keypoints_with_scores):
    """Check if the form for leg raises exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_hip_index = 11
    right_hip_index = 12
    left_knee_index = 13
    right_knee_index = 14
    left_ankle_index = 15
    right_ankle_index = 16

    # Extract keypoint coordinates
    left_hip = keypoints_with_scores[0, 0, left_hip_index, :2]  # x, y coordinates
    right_hip = keypoints_with_scores[0, 0, right_hip_index, :2]
    left_knee = keypoints_with_scores[0, 0, left_knee_index, :2]
    right_knee = keypoints_with_scores[0, 0, right_knee_index, :2]
    left_ankle = keypoints_with_scores[0, 0, left_ankle_index, :2]
    right_ankle = keypoints_with_scores[0, 0, right_ankle_index, :2]

    # Check the alignment of knees with hips
    knee_hip_alignment_threshold = 50  # Adjust as needed
    left_knee_hip_dist = np.linalg.norm(left_knee - left_hip)
    right_knee_hip_dist = np.linalg.norm(right_knee - right_hip)
    if left_knee_hip_dist > knee_hip_alignment_threshold or right_knee_hip_dist > knee_hip_alignment_threshold:
        feedback.append("Align your knees with your hips properly.")

    # Check the angle between lower leg and upper leg
    leg_angle_threshold = 120  # Adjust as needed
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if left_leg_angle < leg_angle_threshold or right_leg_angle < leg_angle_threshold:
        feedback.append("Keep your legs at an appropriate angle to your body.")

    return feedback

import numpy as np

def check_plank(keypoints_with_scores):
    """Check if the form for plank exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    keypoint_indices = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }

    # Extract keypoint coordinates
    keypoints = {name: keypoints_with_scores[0, 0, idx, :2] for name, idx in keypoint_indices.items()}
    
    # Check if the keypoints' scores are reliable
    score_threshold = 0.5
    

    # Helper function to calculate the vertical distance between two points
    def vertical_distance(point1, point2):
        return abs(point1[1] - point2[1])

    # Check shoulder, elbow, and wrist alignment (horizontal alignment)
    def check_horizontal_alignment(keypoints, part1, part2, part3, threshold, side):
        part1_point = keypoints[part1]
        part2_point = keypoints[part2]
        part3_point = keypoints[part3]
        if (abs(part1_point[0] - part2_point[0]) > threshold or 
            abs(part2_point[0] - part3_point[0]) > threshold):
            feedback.append(f"Align your {side} shoulder, elbow, and wrist properly.")

    alignment_threshold = 30
    check_horizontal_alignment(keypoints, 'left_shoulder', 'left_elbow', 'left_wrist', alignment_threshold, 'left')
    check_horizontal_alignment(keypoints, 'right_shoulder', 'right_elbow', 'right_wrist', alignment_threshold, 'right')

    # Check hip and shoulder alignment (vertical alignment)
    hip_shoulder_alignment_threshold = 50
    if vertical_distance(keypoints['left_hip'], keypoints['left_shoulder']) > hip_shoulder_alignment_threshold:
        feedback.append("Keep your left hip aligned with your left shoulder.")
    if vertical_distance(keypoints['right_hip'], keypoints['right_shoulder']) > hip_shoulder_alignment_threshold:
        feedback.append("Keep your right hip aligned with your right shoulder.")

    # Check body straightness by ensuring hips are not sagging or too high
    def check_body_straightness(keypoints, side):
        shoulder_hip_y_diff = keypoints[f'{side}_shoulder'][1] - keypoints[f'{side}_hip'][1]
        hip_ankle_y_diff = keypoints[f'{side}_hip'][1] - keypoints[f'{side}_ankle'][1]
        if shoulder_hip_y_diff > 20:
            feedback.append(f"Your {side} hip is too low; raise it to form a straight line from shoulder to ankle.")
        elif hip_ankle_y_diff > 20:
            feedback.append(f"Your {side} hip is too high; lower it to form a straight line from shoulder to ankle.")

    check_body_straightness(keypoints, 'left')
    check_body_straightness(keypoints, 'right')

    return feedback


def check_pull_up(keypoints_with_scores):
    """Check if the form for pull up exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_elbow_index = 7
    right_elbow_index = 8
    left_wrist_index = 9
    right_wrist_index = 10

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_elbow = keypoints_with_scores[0, 0, left_elbow_index, :2]
    right_elbow = keypoints_with_scores[0, 0, right_elbow_index, :2]
    left_wrist = keypoints_with_scores[0, 0, left_wrist_index, :2]
    right_wrist = keypoints_with_scores[0, 0, right_wrist_index, :2]

    # Check alignment of shoulders, elbows, and wrists
    shoulder_elbow_wrist_alignment_threshold = 30  # Adjust as needed
    left_shoulder_elbow_dist = np.linalg.norm(left_shoulder - left_elbow)
    right_shoulder_elbow_dist = np.linalg.norm(right_shoulder - right_elbow)
    left_elbow_wrist_dist = np.linalg.norm(left_elbow - left_wrist)
    right_elbow_wrist_dist = np.linalg.norm(right_elbow - right_wrist)
    if (left_shoulder_elbow_dist > shoulder_elbow_wrist_alignment_threshold or
        right_shoulder_elbow_dist > shoulder_elbow_wrist_alignment_threshold or
        left_elbow_wrist_dist > shoulder_elbow_wrist_alignment_threshold or
        right_elbow_wrist_dist > shoulder_elbow_wrist_alignment_threshold):
        feedback.append("Align your shoulders, elbows, and wrists properly.")

    return feedback

def check_push_up(keypoints_with_scores):
    """Check if the form for push up exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_elbow_index = 7
    right_elbow_index = 8
    left_wrist_index = 9
    right_wrist_index = 10

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_elbow = keypoints_with_scores[0, 0, left_elbow_index, :2]
    right_elbow = keypoints_with_scores[0, 0, right_elbow_index, :2]
    left_wrist = keypoints_with_scores[0, 0, left_wrist_index, :2]
    right_wrist = keypoints_with_scores[0, 0, right_wrist_index, :2]

    # Check alignment of shoulders, elbows, and wrists
    shoulder_elbow_wrist_alignment_threshold = 30  # Adjust as needed
    left_shoulder_elbow_dist = np.linalg.norm(left_shoulder - left_elbow)
    right_shoulder_elbow_dist = np.linalg.norm(right_shoulder - right_elbow)
    left_elbow_wrist_dist = np.linalg.norm(left_elbow - left_wrist)
    right_elbow_wrist_dist = np.linalg.norm(right_elbow - right_wrist)
    if (left_shoulder_elbow_dist > shoulder_elbow_wrist_alignment_threshold or
        right_shoulder_elbow_dist > shoulder_elbow_wrist_alignment_threshold or
        left_elbow_wrist_dist > shoulder_elbow_wrist_alignment_threshold or
        right_elbow_wrist_dist > shoulder_elbow_wrist_alignment_threshold):
        feedback.append("Align your shoulders, elbows, and wrists properly.")

    return feedback

def check_romanian_deadlift(keypoints_with_scores):
    """Check if the form for Romanian deadlift exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_hip_index = 11
    right_hip_index = 12
    left_knee_index = 13
    right_knee_index = 14
    left_ankle_index = 15
    right_ankle_index = 16

    # Extract keypoint coordinates
    left_hip = keypoints_with_scores[0, 0, left_hip_index, :2]  # x, y coordinates
    right_hip = keypoints_with_scores[0, 0, right_hip_index, :2]
    left_knee = keypoints_with_scores[0, 0, left_knee_index, :2]
    right_knee = keypoints_with_scores[0, 0, right_knee_index, :2]
    left_ankle = keypoints_with_scores[0, 0, left_ankle_index, :2]
    right_ankle = keypoints_with_scores[0, 0, right_ankle_index, :2]

    # Check knee and hip alignment
    knee_hip_alignment_threshold = 30  # Adjust as needed
    left_knee_hip_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_hip_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if (left_knee_hip_angle < knee_hip_alignment_threshold or
        right_knee_hip_angle < knee_hip_alignment_threshold):
        feedback.append("Keep your knees aligned with your hips.")

    return feedback

def check_russian_twist(keypoints_with_scores):
    """Check if the form for Russian twist exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_wrist_index = 9
    right_wrist_index = 10

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_wrist = keypoints_with_scores[0, 0, left_wrist_index, :2]
    right_wrist = keypoints_with_scores[0, 0, right_wrist_index, :2]

    # Check wrist alignment with shoulders
    wrist_shoulder_alignment_threshold = 30  # Adjust as needed
    left_wrist_shoulder_angle = calculate_angle(left_shoulder, left_wrist, right_wrist)
    right_wrist_shoulder_angle = calculate_angle(right_shoulder, right_wrist, left_wrist)
    if (left_wrist_shoulder_angle < wrist_shoulder_alignment_threshold or
        right_wrist_shoulder_angle < wrist_shoulder_alignment_threshold):
        feedback.append("Keep your wrists aligned with your shoulders.")

    return feedback

def check_shoulder_press(keypoints_with_scores):
    """Check if the form for shoulder press exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_elbow_index = 7
    right_elbow_index = 8
    left_wrist_index = 9
    right_wrist_index = 10

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_elbow = keypoints_with_scores[0, 0, left_elbow_index, :2]
    right_elbow = keypoints_with_scores[0, 0, right_elbow_index, :2]
    left_wrist = keypoints_with_scores[0, 0, left_wrist_index, :2]
    right_wrist = keypoints_with_scores[0, 0, right_wrist_index, :2]

    # Check elbow alignment with shoulders
    elbow_shoulder_alignment_threshold = 30  # Adjust as needed
    left_elbow_shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_shoulder_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if (left_elbow_shoulder_angle < elbow_shoulder_alignment_threshold or
        right_elbow_shoulder_angle < elbow_shoulder_alignment_threshold):
        feedback.append("Keep your elbows aligned with your shoulders.")

    return feedback

def check_t_bar_row(keypoints_with_scores):
    """Check if the form for T-Bar Row exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_elbow_index = 7
    right_elbow_index = 8
    left_wrist_index = 9
    right_wrist_index = 10
    hip_index = 11

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_elbow = keypoints_with_scores[0, 0, left_elbow_index, :2]
    right_elbow = keypoints_with_scores[0, 0, right_elbow_index, :2]
    left_wrist = keypoints_with_scores[0, 0, left_wrist_index, :2]
    right_wrist = keypoints_with_scores[0, 0, right_wrist_index, :2]
    hip = keypoints_with_scores[0, 0, hip_index, :2]

    # Check elbow alignment with shoulders
    elbow_shoulder_alignment_threshold = 30  # Adjust as needed
    left_elbow_shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_shoulder_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if (left_elbow_shoulder_angle < elbow_shoulder_alignment_threshold or
        right_elbow_shoulder_angle < elbow_shoulder_alignment_threshold):
        feedback.append("Keep your elbows aligned with your shoulders.")

    # Check hip alignment with shoulders
    hip_shoulder_alignment_threshold = 30  # Adjust as needed
    left_hip_shoulder_angle = calculate_angle(left_shoulder, hip, left_elbow)
    right_hip_shoulder_angle = calculate_angle(right_shoulder, hip, right_elbow)
    if (left_hip_shoulder_angle < hip_shoulder_alignment_threshold or
        right_hip_shoulder_angle < hip_shoulder_alignment_threshold):
        feedback.append("Keep your hips aligned with your shoulders.")

    return feedback

def check_tricep_dips(keypoints_with_scores):
    """Check if the form for tricep dips exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_elbow_index = 7
    right_elbow_index = 8
    left_wrist_index = 9
    right_wrist_index = 10

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_elbow = keypoints_with_scores[0, 0, left_elbow_index, :2]
    right_elbow = keypoints_with_scores[0, 0, right_elbow_index, :2]
    left_wrist = keypoints_with_scores[0, 0, left_wrist_index, :2]
    right_wrist = keypoints_with_scores[0, 0, right_wrist_index, :2]

    # Check elbow alignment with shoulders
    elbow_shoulder_alignment_threshold = 30  # Adjust as needed
    left_elbow_shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_shoulder_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if (left_elbow_shoulder_angle < elbow_shoulder_alignment_threshold or
        right_elbow_shoulder_angle < elbow_shoulder_alignment_threshold):
        feedback.append("Keep your elbows aligned with your shoulders.")

    return feedback



def check_tricep_pushdown(keypoints_with_scores):
    """Check if the form for tricep pushdown exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_shoulder_index = 5
    right_shoulder_index = 6
    left_elbow_index = 7
    right_elbow_index = 8
    left_wrist_index = 9
    right_wrist_index = 10

    # Extract keypoint coordinates
    left_shoulder = keypoints_with_scores[0, 0, left_shoulder_index, :2]  # x, y coordinates
    right_shoulder = keypoints_with_scores[0, 0, right_shoulder_index, :2]
    left_elbow = keypoints_with_scores[0, 0, left_elbow_index, :2]
    right_elbow = keypoints_with_scores[0, 0, right_elbow_index, :2]
    left_wrist = keypoints_with_scores[0, 0, left_wrist_index, :2]
    right_wrist = keypoints_with_scores[0, 0, right_wrist_index, :2]

    # Check elbow alignment with shoulders
    elbow_shoulder_alignment_threshold = 30  # Adjust as needed
    left_elbow_shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_shoulder_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if (left_elbow_shoulder_angle < elbow_shoulder_alignment_threshold or
        right_elbow_shoulder_angle < elbow_shoulder_alignment_threshold):
        feedback.append("Keep your elbows aligned with your shoulders.")

    return feedback

def check_squat(keypoints_with_scores):
    """Check if the form for squat exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_hip_index = 11
    right_hip_index = 12
    left_knee_index = 13
    right_knee_index = 14
    left_ankle_index = 15
    right_ankle_index = 16

    # Extract keypoint coordinates
    left_hip = keypoints_with_scores[0, 0, left_hip_index, :2]  # x, y coordinates
    right_hip = keypoints_with_scores[0, 0, right_hip_index, :2]
    left_knee = keypoints_with_scores[0, 0, left_knee_index, :2]
    right_knee = keypoints_with_scores[0, 0, right_knee_index, :2]
    left_ankle = keypoints_with_scores[0, 0, left_ankle_index, :2]
    right_ankle = keypoints_with_scores[0, 0, right_ankle_index, :2]

    # Check knee and hip alignment
    knee_hip_alignment_threshold = 30  # Adjust as needed
    left_knee_hip_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_hip_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if (left_knee_hip_angle < knee_hip_alignment_threshold or
        right_knee_hip_angle < knee_hip_alignment_threshold):
        feedback.append("Keep your knees aligned with your hips.")

    return feedback


def check_deadlift(keypoints_with_scores):
    """Check if the form for a deadlift exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Extract keypoint coordinates
    kpts_x = keypoints_with_scores[0, 0, :, 1]
    kpts_y = keypoints_with_scores[0, 0, :, 0]
    kpts_scores = keypoints_with_scores[0, 0, :, 2]

    # Check alignment of hips, knees, and ankles
    left_hip = (kpts_x[KEYPOINT_DICT['left_hip']], kpts_y[KEYPOINT_DICT['left_hip']])
    right_hip = (kpts_x[KEYPOINT_DICT['right_hip']], kpts_y[KEYPOINT_DICT['right_hip']])
    left_knee = (kpts_x[KEYPOINT_DICT['left_knee']], kpts_y[KEYPOINT_DICT['left_knee']])
    right_knee = (kpts_x[KEYPOINT_DICT['right_knee']], kpts_y[KEYPOINT_DICT['right_knee']])
    left_ankle = (kpts_x[KEYPOINT_DICT['left_ankle']], kpts_y[KEYPOINT_DICT['left_ankle']])
    right_ankle = (kpts_x[KEYPOINT_DICT['right_ankle']], kpts_y[KEYPOINT_DICT['right_ankle']])
    
    left_hip_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_hip_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    if left_hip_knee_angle < 150 or right_hip_knee_angle < 150:
        feedback.append("Hips and knees are not properly aligned.")
    else:
        feedback.append("Hips and knees are properly aligned.")

    # Check if shoulders are aligned with hips
    left_shoulder = (kpts_x[KEYPOINT_DICT['left_shoulder']], kpts_y[KEYPOINT_DICT['left_shoulder']])
    right_shoulder = (kpts_x[KEYPOINT_DICT['right_shoulder']], kpts_y[KEYPOINT_DICT['right_shoulder']])
    
    left_shoulder_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_shoulder_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    
    if left_shoulder_hip_angle < 150 or right_shoulder_hip_angle < 150:
        feedback.append("Shoulders and hips are not properly aligned.")
    else:
        feedback.append("Shoulders and hips are properly aligned.")

    return feedback

def check_squat(keypoints_with_scores):
    """Check if the form for squat exercise is correct based on keypoints.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
            the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
        A list of strings containing feedback messages noting the issues present
        in the form, if any.
    """
    feedback = []

    # Define the indices of relevant keypoints
    left_hip_index = 11
    right_hip_index = 12
    left_knee_index = 13
    right_knee_index = 14
    left_ankle_index = 15
    right_ankle_index = 16

    # Extract keypoint coordinates
    left_hip = keypoints_with_scores[0, 0, left_hip_index, :2]  # x, y coordinates
    right_hip = keypoints_with_scores[0, 0, right_hip_index, :2]
    left_knee = keypoints_with_scores[0, 0, left_knee_index, :2]
    right_knee = keypoints_with_scores[0, 0, right_knee_index, :2]
    left_ankle = keypoints_with_scores[0, 0, left_ankle_index, :2]
    right_ankle = keypoints_with_scores[0, 0, right_ankle_index, :2]

    # Check knee and hip alignment
    knee_hip_alignment_threshold = 30  # Adjust as needed
    left_knee_hip_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_hip_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if (left_knee_hip_angle < knee_hip_alignment_threshold or
        right_knee_hip_angle < knee_hip_alignment_threshold):
        feedback.append("Keep your knees aligned with your hips.")

    return feedback
