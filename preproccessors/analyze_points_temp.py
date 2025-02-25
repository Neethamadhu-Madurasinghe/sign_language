import json
import numpy as np
import matplotlib.pyplot as plt

# Load left hand landmarks
with open("left_hand_landmarks.json", "r") as f:
    left_hand_landmarks = json.load(f)[0]  # Assuming first hand in list

# Mirror left hand landmarks to create right hand
right_hand_landmarks = []
for point in left_hand_landmarks:
    mirrored_point = {
        "id": point["id"],
        "x": 1 - point["x"],  # Flip horizontally
        "y": point["y"],
        "z": point["z"]
    }
    right_hand_landmarks.append(mirrored_point)

# Define pose skeleton manually (dummy data, replace with actual pose landmarks)
pose_landmarks = [
    {"id": i, "x": np.random.rand(), "y": np.random.rand(), "z": 0} for i in range(33)
]

# Define skeleton connections (MediaPipe Pose)
pose_connections = [(16, 18), (18, 20), (20, 22), (15, 17), (17, 19), (19, 21), (15, 30), (30, 32)]
hand_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                     (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
                     (13, 17), (17, 18), (18, 19), (19, 20)]

# Highlighted keypoints
pose_highlight = [18, 20, 22, 16, 19, 17, 21, 15, 30, 32]
left_hand_highlight = [8, 12, 16, 7, 11, 20, 15, 4, 19, 10]
right_hand_highlight = [8, 12, 7, 16, 11, 4, 20, 15, 6, 19]

def plot_landmarks(ax, landmarks, connections, highlight, title, color='blue'):
    x = [p['x'] for p in landmarks]
    y = [p['y'] for p in landmarks]
    ax.scatter(x, y, color=color)
    
    for (i, j) in connections:
        ax.plot([landmarks[i]['x'], landmarks[j]['x']],
                [landmarks[i]['y'], landmarks[j]['y']], color=color)
    
    for idx in highlight:
        ax.scatter(landmarks[idx]['x'], landmarks[idx]['y'], color='red', s=50)
    
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])

# Plot pose, left, and right hand
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_landmarks(axs[0], pose_landmarks, pose_connections, pose_highlight, "Pose", color='green')
plot_landmarks(axs[1], left_hand_landmarks, hand_connections, left_hand_highlight, "Left Hand", color='blue')
plot_landmarks(axs[2], right_hand_landmarks, hand_connections, right_hand_highlight, "Right Hand", color='purple')

plt.show()
