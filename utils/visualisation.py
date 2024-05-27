
def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors

#@title Helper functions for visualization

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

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}



def draw_prediction_on_image(image, keypoints_with_scores, feedback=None, crop_region=None, close_figure=False, output_image_height=None):
    """Draws the keypoint predictions on image.

    Args:
        image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        feedback: A list of strings representing feedback messages.
        crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
        output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
        A numpy array with shape [out_height, out_width, channel] representing the
        image overlaid with keypoint predictions and feedback messages.
    """
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin,ymin),rec_width,rec_height,
            linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)

    if feedback:
        feedback_text = '\n'.join(feedback)
        plt.text(0, -20, feedback_text, fontsize=12, color='red')

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC)
    return image_from_plot

def to_gif(images, duration):
  """Converts image sequence (4D numpy array) to gif."""
  imageio.mimsave('./animation.gif', images, duration=duration)
  return embed.embed_file('./animation.gif')

def progress(value, max=100):
  return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))

  import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio
from IPython.display import HTML, display

def draw_prediction_on_image(image, keypoints_with_scores, feedback=None, crop_region=None, close_figure=False, output_image_height=None):
    """Draws the keypoint predictions on image.

    Args:
        image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        feedback: A list of strings representing feedback messages. The list can
        contain both positive and negative feedback.
        crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
        output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
        A numpy array with shape [out_height, out_width, channel] representing the
        image overlaid with keypoint predictions and feedback messages.
    """
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin,ymin),rec_width,rec_height,
            linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)

    if feedback:
        feedback_text = '\n'.join(feedback)
        plt.text(0, 20, feedback_text, fontsize=20, color='red')

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC)
    return image_from_plot

def check_squat_front_facing(keypoints_with_scores):
    """Check if the form for a squat exercise is correct based on keypoints.

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

    # Check if left and right knees are detected
    left_knee_score = kpts_scores[11]
    right_knee_score = kpts_scores[12]
    if left_knee_score < 0.1 or right_knee_score < 0.1:
        feedback.append("Both knees are not detected.")
    else:
        feedback.append("Knees are properly detected.")

    # Check if hips are detected
    left_hip_score = kpts_scores[5]
    right_hip_score = kpts_scores[6]
    if left_hip_score < 0.1 or right_hip_score < 0.1:
        feedback.append("Both hips are not detected.")
    else:
        feedback.append("Hips are properly detected.")

    # Check if shoulders are detected
    left_shoulder_score = kpts_scores[11]
    right_shoulder_score = kpts_scores[12]
    if left_shoulder_score < 0.1 or right_shoulder_score < 0.1:
        feedback.append("Both shoulders are not detected.")
    else:
        feedback.append("Shoulders are properly detected.")

    # Check if ankles are detected
    left_ankle_score = kpts_scores[13]
    right_ankle_score = kpts_scores[14]
    if left_ankle_score < 0.1 or right_ankle_score < 0.1:
        feedback.append("Both ankles are not detected.")
    else:
        feedback.append("Ankles are properly detected.")

    # Check if knees are aligned with toes
    left_knee_x = kpts_x[11]
    right_knee_x = kpts_x[12]
    left_ankle_x = kpts_x[13]
    right_ankle_x = kpts_x[14]
    if abs(left_knee_x - left_ankle_x) > 50 or abs(right_knee_x - right_ankle_x) > 50:
        feedback.append("Knees are not aligned with toes.")
    else:
        feedback.append("Knees are aligned with toes.")

    return feedback

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

def check_squat_front_facing(keypoints_with_scores):
    """Check if the form for a squat exercise is correct based on keypoints.

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

    # Check if knees are detected
    left_knee_score = kpts_scores[KEYPOINT_DICT['left_knee']]
    right_knee_score = kpts_scores[KEYPOINT_DICT['right_knee']]
    if left_knee_score < 0.1 or right_knee_score < 0.1:
        feedback.append("Both knees are not detected.")
   

    # Check if hips are detected
    left_hip_score = kpts_scores[KEYPOINT_DICT['left_hip']]
    right_hip_score = kpts_scores[KEYPOINT_DICT['right_hip']]
    if left_hip_score < 0.1 or right_hip_score < 0.1:
        feedback.append("Both hips are not detected.")
    
    # Check if ankles are detected
    left_ankle_score = kpts_scores[KEYPOINT_DICT['left_ankle']]
    right_ankle_score = kpts_scores[KEYPOINT_DICT['right_ankle']]
    if left_ankle_score < 0.1 or right_ankle_score < 0.1:
        feedback.append("Both ankles are not detected.")
    

    # Check if knees are aligned with toes
    left_knee = (kpts_x[KEYPOINT_DICT['left_knee']], kpts_y[KEYPOINT_DICT['left_knee']])
    right_knee = (kpts_x[KEYPOINT_DICT['right_knee']], kpts_y[KEYPOINT_DICT['right_knee']])
    left_ankle = (kpts_x[KEYPOINT_DICT['left_ankle']], kpts_y[KEYPOINT_DICT['left_ankle']])
    right_ankle = (kpts_x[KEYPOINT_DICT['right_ankle']], kpts_y[KEYPOINT_DICT['right_ankle']])
    left_hip = (kpts_x[KEYPOINT_DICT['left_hip']], kpts_y[KEYPOINT_DICT['left_hip']])
    right_hip = (kpts_x[KEYPOINT_DICT['right_hip']], kpts_y[KEYPOINT_DICT['right_hip']])

    
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    if left_knee_angle < 150 or right_knee_angle < 150:
        feedback.append("Knees are not properly aligned with the toes.")
    else:
        feedback.append("Knees are properly aligned with the toes.")

    # Check if knees are in line with hips
    
    left_hip_knee_angle = calculate_angle(left_hip, left_knee, right_knee)
    right_hip_knee_angle = calculate_angle(right_hip, right_knee, left_knee)
    if left_hip_knee_angle < 150 or right_hip_knee_angle < 150:
        feedback.append("Hips and knees are not properly aligned.")
    else:
        feedback.append("Hips and knees are properly aligned.")

    return feedback



