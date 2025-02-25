import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the pickle file
def load_single_video(file_path):
    with open(file_path, 'rb') as f:
        landmarks = pickle.load(f)
    return landmarks

# Analyze movement
def analyze_movement(landmarks, verbose=False):
    pose_movement = np.zeros(33)
    left_hand_movement = np.zeros(21)
    right_hand_movement = np.zeros(21)
    
    for i in range(1, len(landmarks)):
        prev_frame = landmarks[i - 1]
        curr_frame = landmarks[i]
        
        pose_prev = prev_frame['pose_landmarks']
        pose_curr = curr_frame['pose_landmarks']
        left_prev = prev_frame['left_hand_landmarks']
        left_curr = curr_frame['left_hand_landmarks']
        right_prev = prev_frame['right_hand_landmarks']
        right_curr = curr_frame['right_hand_landmarks']
        
        if pose_prev is not None and pose_curr is not None:
            for idx in range(33):
                if pose_prev[idx] and pose_curr[idx]:
                    dx = pose_curr[idx]['x'] - pose_prev[idx]['x']
                    dy = pose_curr[idx]['y'] - pose_prev[idx]['y']
                    distance = np.sqrt(dx**2 + dy**2)
                    pose_movement[idx] += distance
        
        if left_prev is not None and left_curr is not None and left_prev[0] and left_curr[0]:
            wrist_dx = left_curr[0]['x'] - left_prev[0]['x']
            wrist_dy = left_curr[0]['y'] - left_prev[0]['y']
            for idx in range(21):
                if left_prev[idx] and left_curr[idx]:
                    dx = left_curr[idx]['x'] - left_prev[idx]['x'] - wrist_dx
                    dy = left_curr[idx]['y'] - left_prev[idx]['y'] - wrist_dy
                    distance = np.sqrt(dx**2 + dy**2)
                    left_hand_movement[idx] += distance
        
        if right_prev is not None and right_curr is not None and right_prev[0] and right_curr[0]:
            wrist_dx = right_curr[0]['x'] - right_prev[0]['x']
            wrist_dy = right_curr[0]['y'] - right_prev[0]['y']
            for idx in range(21):
                if right_prev[idx] and right_curr[idx]:
                    dx = right_curr[idx]['x'] - right_prev[idx]['x'] - wrist_dx
                    dy = right_curr[idx]['y'] - right_prev[idx]['y'] - wrist_dy
                    distance = np.sqrt(dx**2 + dy**2)
                    right_hand_movement[idx] += distance
    
    if verbose:
        print("Pose movement sum:", pose_movement.sum())
        print("Left hand movement sum:", left_hand_movement.sum())
        print("Right hand movement sum:", right_hand_movement.sum())
    
    return pose_movement, left_hand_movement, right_hand_movement

# Combined plot function
def plot_combined(landmarks, pose_movement, left_hand_movement, right_hand_movement, file_path="Unknown"):
    # Frame selection: prioritize right hand, then pose, then first frame
    frame = next((f for f in landmarks if f['right_hand_landmarks'] is not None and any(f['right_hand_landmarks'])), 
                 next((f for f in landmarks if f['pose_landmarks'] is not None), landmarks[0]))
    frame_idx = landmarks.index(frame)
    print(f"File: {file_path}")
    print(f"Selected frame index: {frame_idx}")
    print(f"Pose landmarks present: {frame['pose_landmarks'] is not None}")
    print(f"Left hand landmarks present: {frame['left_hand_landmarks'] is not None}")
    print(f"Right hand landmarks present: {frame['right_hand_landmarks'] is not None}")
    
    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)
    ]
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    # 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), width_ratios=[2, 1.5, 1.5], height_ratios=[1, 1.5])
    fig.suptitle(f"Analysis for {file_path.split('/')[-1]}", fontsize=16)
    
    # Add unified scale calculation before plotting:
    global_max = max(left_hand_movement.max(), right_hand_movement.max(), 0.001)

    # Update bar charts with unified y-axis scale:
    axes[0, 0].bar(range(33), pose_movement)
    axes[0, 0].set_title("Pose Movement")
    axes[0, 0].set_xlabel("Landmark Index")
    axes[0, 0].set_ylabel("Total Movement")

    axes[0, 1].bar(range(21), left_hand_movement)
    axes[0, 1].set_title("Left Hand Movement")
    axes[0, 1].set_xlabel("Landmark Index")
    axes[0, 1].set_ylim(0, global_max * 1.1)  # Added

    axes[0, 2].bar(range(21), right_hand_movement)
    axes[0, 2].set_title("Right Hand Movement")
    axes[0, 2].set_xlabel("Landmark Index")
    axes[0, 2].set_ylim(0, global_max * 1.1)  # Added
    
    # Row 2: Skeletons
    # Pose skeleton
    if frame['pose_landmarks'] is not None:
        pose_scatter = axes[1, 0].scatter(
            [p['x'] for p in frame['pose_landmarks'] if p],
            [p['y'] for p in frame['pose_landmarks'] if p],
            c=[pose_movement[i] for i, p in enumerate(frame['pose_landmarks']) if p],
            cmap='hot', vmin=0, vmax=max(pose_movement.max(), 0.001)
        )
        for start, end in pose_connections:
            if frame['pose_landmarks'][start] and frame['pose_landmarks'][end]:
                axes[1, 0].plot([frame['pose_landmarks'][start]['x'], frame['pose_landmarks'][end]['x']],
                                [frame['pose_landmarks'][start]['y'], frame['pose_landmarks'][end]['y']], 'gray')
        plt.colorbar(pose_scatter, ax=axes[1, 0], label="Movement")
    axes[1, 0].set_title("Pose Skeleton")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_aspect('equal')
    
    # Left hand skeleton
    if frame['left_hand_landmarks'] is not None:
        left_valid = [p for p in frame['left_hand_landmarks'] if p]
        if left_valid:
            left_scatter = axes[1, 1].scatter(
                [p['x'] for p in left_valid],
                [p['y'] for p in left_valid],
                c=[left_hand_movement[i] for i, p in enumerate(frame['left_hand_landmarks']) if p],
                cmap='hot', vmin=0, vmax=global_max
            )
            for start, end in hand_connections:
                if frame['left_hand_landmarks'][start] and frame['left_hand_landmarks'][end]:
                    axes[1, 1].plot([frame['left_hand_landmarks'][start]['x'], frame['left_hand_landmarks'][end]['x']],
                                    [frame['left_hand_landmarks'][start]['y'], frame['left_hand_landmarks'][end]['y']], 'gray')
            plt.colorbar(left_scatter, ax=axes[1, 1], label="Movement")
        else:
            print("Left hand: No valid points")
    else:
        print("Left hand: No landmarks")
    axes[1, 1].set_title("Left Hand Skeleton")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_aspect('equal')
    
    # Right hand skeleton - Fixed typo (was plotting on left hand axes)
    if frame['right_hand_landmarks'] is not None:
        right_valid = [p for p in frame['right_hand_landmarks'] if p]
        if right_valid:
            right_scatter = axes[1, 2].scatter(
                [p['x'] for p in right_valid],
                [p['y'] for p in right_valid],
                c=[right_hand_movement[i] for i, p in enumerate(frame['right_hand_landmarks']) if p],
                cmap='hot', vmin=0, vmax=global_max
            )
            for start, end in hand_connections:
                if frame['right_hand_landmarks'][start] and frame['right_hand_landmarks'][end]:
                    axes[1, 2].plot([frame['right_hand_landmarks'][start]['x'], frame['right_hand_landmarks'][end]['x']],
                                    [frame['right_hand_landmarks'][start]['y'], frame['right_hand_landmarks'][end]['y']], 'gray')
            plt.colorbar(right_scatter, ax=axes[1, 2], label="Movement")
        else:
            print("Right hand: No valid points")
    else:
        print("Right hand: No landmarks")
    axes[1, 2].set_title("Right Hand Skeleton")
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_aspect('equal')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.show()

# Main execution
file_path = '../all_outputs/output_ssl_small/w003/2-1-w00320221110095852.mp4_1.pkl'  # Your file
file_path = '../all_outputs/output_ssl_small/w022/1-2-w02220221103121901.mp4_1.pkl'  # Your file

landmarks = load_single_video(file_path)
pose_movement, left_hand_movement, right_hand_movement = analyze_movement(landmarks, verbose=True)

print("Top 5 Pose Landmarks by Movement:", np.argsort(pose_movement)[-5:][::-1])
print("Top 5 Left Hand Landmarks by Movement:", np.argsort(left_hand_movement)[-5:][::-1])
print("Top 5 Right Hand Landmarks by Movement:", np.argsort(right_hand_movement)[-5:][::-1])

plot_combined(landmarks, pose_movement, left_hand_movement, right_hand_movement, file_path)