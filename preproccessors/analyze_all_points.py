

# For a gieven pickle file set this code will check the movement of skeleton points on average



import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

def load_single_video(file_path):
    with open(file_path, 'rb') as f:
        landmarks = pickle.load(f)
    return landmarks

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

def plot_combined(frame, pose_movement, left_hand_movement, right_hand_movement, file_path="Unknown", save_path=None):
    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),  # Face outline (nose to ears)
        (9, 10),  # Mouth
        (11, 12),  # Shoulders
        (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),  # Left arm (shoulder to wrist to fingers)
        (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),  # Right arm
        (11, 23), (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg (hip to ankle to toes)
        (12, 24), (24, 26), (26, 28), (28, 30), (30, 32)   # Right leg
    ]
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (0, 5), (0, 9), (0, 13), (0, 17)  # Wrist to MCP of each finger
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), width_ratios=[2, 1.5, 1.5], height_ratios=[1, 1.5])
    fig.suptitle(f"Analysis for {file_path}", fontsize=16)
    
    global_max = max(left_hand_movement.max(), right_hand_movement.max(), pose_movement.max(), 0.001)

    # Bar charts
    axes[0, 0].bar(range(33), pose_movement)
    axes[0, 0].set_title("Pose Movement")
    axes[0, 0].set_xlabel("Landmark Index")
    axes[0, 0].set_ylabel("Avg Movement per Interval")
    axes[0, 0].set_ylim(0, global_max * 1.1)
    
    axes[0, 1].bar(range(21), left_hand_movement)
    axes[0, 1].set_title("Left Hand Movement")
    axes[0, 1].set_xlabel("Landmark Index")
    axes[0, 1].set_ylim(0, global_max * 1.1)
    
    axes[0, 2].bar(range(21), right_hand_movement)
    axes[0, 2].set_title("Right Hand Movement")
    axes[0, 2].set_xlabel("Landmark Index")
    axes[0, 2].set_ylim(0, global_max * 1.1)
    
    # Skeletons with debug prints
    # Pose skeleton (frame-based, plot if pose_landmarks exist)
    if frame and frame['pose_landmarks'] is not None:
        pose_valid = [p for p in frame['pose_landmarks'] if p and isinstance(p, dict) and 'x' in p and 'y' in p]
        if pose_valid:
            pose_scatter = axes[1, 0].scatter(
                [p['x'] for p in pose_valid],
                [p['y'] for p in pose_valid],
                c=[pose_movement[i] for i, p in enumerate(frame['pose_landmarks']) if p and isinstance(p, dict) and 'x' in p and 'y' in p],
                cmap='hot', vmin=0, vmax=global_max
            )
            for start, end in pose_connections:
                if (frame['pose_landmarks'][start] and frame['pose_landmarks'][end] and 
                    isinstance(frame['pose_landmarks'][start], dict) and 'x' in frame['pose_landmarks'][start] and 
                    isinstance(frame['pose_landmarks'][end], dict) and 'x' in frame['pose_landmarks'][end]):
                    axes[1, 0].plot([frame['pose_landmarks'][start]['x'], frame['pose_landmarks'][end]['x']],
                                    [frame['pose_landmarks'][start]['y'], frame['pose_landmarks'][end]['y']], 'gray', linewidth=1.5)
            plt.colorbar(pose_scatter, ax=axes[1, 0], label="Movement")
        else:
            print(f"{file_path}: No valid pose points in frame")
    else:
        print(f"{file_path}: No pose landmarks available")
    axes[1, 0].set_title("Pose Skeleton")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_aspect('equal')
    
    # Left hand skeleton (using actual MediaPipe coords, plot only if left_hand_movement has non-zero values)
    if np.any(left_hand_movement > 0):
        # Load actual left hand landmarks from a JSON file
        try:
            with open("left_hand_landmarks.json", "r") as f:
                left_hand_landmarks = json.load(f)
            left_hand = np.array([(lm["x"], lm["y"]) for lm in left_hand_landmarks[0]])
        except FileNotFoundError:
            print(f"Warning: left_hand_landmarks.json not found for {file_path}, using default coordinates")
            # Fallback default coordinates if JSON is missing
            left_hand = np.array([
                [0.3, 0.2], [0.35, 0.3], [0.4, 0.4], [0.45, 0.5], [0.5, 0.6],  # Thumb
                [0.3, 0.3], [0.35, 0.4], [0.4, 0.5], [0.45, 0.6],  # Index
                [0.3, 0.35], [0.35, 0.45], [0.4, 0.55], [0.45, 0.65],  # Middle
                [0.3, 0.4], [0.35, 0.5], [0.4, 0.6], [0.45, 0.7],  # Ring
                [0.3, 0.45], [0.35, 0.55], [0.4, 0.65], [0.45, 0.75]  # Pinky
            ])
        
        valid_indices = [i for i in range(21) if left_hand_movement[i] > 0]
        if valid_indices:
            x_coords = [left_hand[i, 0] for i in valid_indices]
            y_coords = [-left_hand[i, 1] for i in valid_indices]  # Invert y for MediaPipe orientation
            movements = [left_hand_movement[i] for i in valid_indices]
            
            scatter = axes[1, 1].scatter(x_coords, y_coords, c=movements, cmap='hot', vmin=0, vmax=global_max, s=50)
            for start, end in hand_connections:
                if start in valid_indices and end in valid_indices:
                    axes[1, 1].plot([left_hand[start, 0], left_hand[end, 0]],
                                    [-left_hand[start, 1], -left_hand[end, 1]], 'gray', linewidth=2)
            plt.colorbar(scatter, ax=axes[1, 1], label="Movement")
            print(f"{file_path}: Left hand valid points: {len(valid_indices)}")
        else:
            print(f"{file_path}: No non-zero left hand movement for plotting")
    else:
        print(f"{file_path}: No left hand movement detected, skeleton not plotted")
    axes[1, 1].set_title("Left Hand Skeleton")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(-1, 0)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    # Right hand skeleton (mirrored, plot only if right_hand_movement has non-zero values)
    if np.any(right_hand_movement > 0):
        # Mirror left hand to get right hand coordinates
        try:
            with open("left_hand_landmarks.json", "r") as f:
                left_hand_landmarks = json.load(f)
            left_hand = np.array([(lm["x"], lm["y"]) for lm in left_hand_landmarks[0]])
            right_hand = left_hand.copy()
            right_hand[:, 0] = 1 - right_hand[:, 0]  # Flip x-coordinates to simulate right hand
        except FileNotFoundError:
            print(f"Warning: left_hand_landmarks.json not found for {file_path}, using default coordinates")
            # Fallback default coordinates if JSON is missing
            left_hand = np.array([
                [0.3, 0.2], [0.35, 0.3], [0.4, 0.4], [0.45, 0.5], [0.5, 0.6],  # Thumb
                [0.3, 0.3], [0.35, 0.4], [0.4, 0.5], [0.45, 0.6],  # Index
                [0.3, 0.35], [0.35, 0.45], [0.4, 0.55], [0.45, 0.65],  # Middle
                [0.3, 0.4], [0.35, 0.5], [0.4, 0.6], [0.45, 0.7],  # Ring
                [0.3, 0.45], [0.35, 0.55], [0.4, 0.65], [0.45, 0.75]  # Pinky
            ])
            right_hand = left_hand.copy()
            right_hand[:, 0] = 1 - right_hand[:, 0]  # Mirror for right hand
        
        valid_indices = [i for i in range(21) if right_hand_movement[i] > 0]
        if valid_indices:
            x_coords = [right_hand[i, 0] for i in valid_indices]
            y_coords = [-right_hand[i, 1] for i in valid_indices]  # Invert y for MediaPipe orientation
            movements = [right_hand_movement[i] for i in valid_indices]
            
            scatter = axes[1, 2].scatter(x_coords, y_coords, c=movements, cmap='hot', vmin=0, vmax=global_max, s=50)
            for start, end in hand_connections:
                if start in valid_indices and end in valid_indices:
                    axes[1, 2].plot([right_hand[start, 0], right_hand[end, 0]],
                                    [-right_hand[start, 1], -right_hand[end, 1]], 'gray', linewidth=2)
            plt.colorbar(scatter, ax=axes[1, 2], label="Movement")
            print(f"{file_path}: Right hand valid points: {len(valid_indices)}")
        else:
            print(f"{file_path}: No non-zero right hand movement for plotting")
    else:
        print(f"{file_path}: No right hand movement detected, skeleton not plotted")
    axes[1, 2].set_title("Right Hand Skeleton")
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(-1, 0)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    plt.close()

def generate_report(main_dir, output_html):
    # Set output directory to 'plots' in the current working directory
    output_dir = os.path.join(".", "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    classes = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d)) and d != "plots"]
    
    all_pose_avg_list = []
    all_left_hand_avg_list = []
    all_right_hand_avg_list = []
    html_content = "<html><body><h1>Movement Analysis Report</h1>"
    first_frame = None
    
    for class_name in classes:
        class_dir = os.path.join(main_dir, class_name)
        pkl_files = [f for f in os.listdir(class_dir) if f.endswith('.pkl')]
        if not pkl_files:
            print(f"No videos in class {class_name}")
            continue
        
        pose_avg_list = []
        left_hand_avg_list = []
        right_hand_avg_list = []
        class_frame = None
        
        for pkl_file in pkl_files:
            file_path = os.path.join(class_dir, pkl_file)
            try:
                landmarks = load_single_video(file_path)
                if len(landmarks) < 2:
                    print(f"Skipping {pkl_file}: not enough frames")
                    continue
                
                # Find a frame with pose and at least one hand, handling NoneType
                class_frame_temp = None
                for frame in landmarks:
                    has_pose = frame['pose_landmarks'] is not None and any(frame['pose_landmarks']) if frame['pose_landmarks'] else False
                    has_left_hand = (frame['left_hand_landmarks'] is not None and 
                                   any(frame['left_hand_landmarks'])) if frame['left_hand_landmarks'] else False
                    has_right_hand = (frame['right_hand_landmarks'] is not None and 
                                    any(frame['right_hand_landmarks'])) if frame['right_hand_landmarks'] else False
                    
                    if has_pose and (has_left_hand or has_right_hand):
                        class_frame_temp = frame
                        break
                
                # Fallback to first frame with pose if no hand data found
                if class_frame_temp is None:
                    class_frame_temp = next((f for f in landmarks if f['pose_landmarks'] is not None), landmarks[0])
                    print(f"Warning: No hand data in {pkl_file} for skeleton plotting, using first pose frame")
                
                if class_frame is None:
                    class_frame = class_frame_temp
                
                if first_frame is None:
                    first_frame_temp = None
                    for frame in landmarks:
                        has_pose = frame['pose_landmarks'] is not None and any(frame['pose_landmarks']) if frame['pose_landmarks'] else False
                        has_left_hand = (frame['left_hand_landmarks'] is not None and 
                                       any(frame['left_hand_landmarks'])) if frame['left_hand_landmarks'] else False
                        has_right_hand = (frame['right_hand_landmarks'] is not None and 
                                        any(frame['right_hand_landmarks'])) if frame['right_hand_landmarks'] else False
                        
                        if has_pose and (has_left_hand or has_right_hand):
                            first_frame_temp = frame
                            break
                    
                    if first_frame_temp is None:
                        first_frame_temp = next((f for f in landmarks if f['pose_landmarks'] is not None), landmarks[0])
                        print(f"Warning: No hand data in {pkl_file} for overall skeleton, using first pose frame")
                    first_frame = first_frame_temp
                
                pose_total, left_hand_total, right_hand_total = analyze_movement(landmarks)
                num_intervals = len(landmarks) - 1
                pose_avg = pose_total / num_intervals if num_intervals > 0 else np.zeros(33)
                left_hand_avg = left_hand_total / num_intervals if num_intervals > 0 else np.zeros(21)
                right_hand_avg = right_hand_total / num_intervals if num_intervals > 0 else np.zeros(21)
                
                pose_avg_list.append(pose_avg)
                left_hand_avg_list.append(left_hand_avg)
                right_hand_avg_list.append(right_hand_avg)
                all_pose_avg_list.append(pose_avg)
                all_left_hand_avg_list.append(left_hand_avg)
                all_right_hand_avg_list.append(right_hand_avg)
            except Exception as e:
                print(f"Error processing {pkl_file}: {e}")
                continue
        
        if pose_avg_list:
            class_pose_avg = np.mean(pose_avg_list, axis=0)
            class_left_hand_avg = np.mean(left_hand_avg_list, axis=0)
            class_right_hand_avg = np.mean(right_hand_avg_list, axis=0)
            
            plot_file = os.path.join(output_dir, f"{class_name}_plots.png")
            plot_combined(class_frame, class_pose_avg, class_left_hand_avg, class_right_hand_avg, 
                          file_path=class_name, save_path=plot_file)
            
            rel_plot_path = os.path.relpath(plot_file, os.path.dirname(output_html))
            html_content += f"<h2>Class: {class_name}</h2>"
            html_content += f'<img src="{rel_plot_path}" alt="{class_name} plots" style="max-width:100%;"><br>'
    
    if all_pose_avg_list:
        overall_pose_avg = np.mean(all_pose_avg_list, axis=0)
        overall_left_hand_avg = np.mean(all_left_hand_avg_list, axis=0)
        overall_right_hand_avg = np.mean(all_right_hand_avg_list, axis=0)
        
        plot_file = os.path.join(output_dir, "overall_plots.png")
        plot_combined(first_frame, overall_pose_avg, overall_left_hand_avg, overall_right_hand_avg, 
                      file_path="Overall", save_path=plot_file)
        
        rel_plot_path = os.path.relpath(plot_file, os.path.dirname(output_html))
        html_content += "<h2>Overall Average Across All Classes</h2>"
        html_content += f'<img src="{rel_plot_path}" alt="Overall plots" style="max-width:100%;"><br>'
        
        top_pose = np.argsort(overall_pose_avg)[::-1][:10]
        top_left_hand = np.argsort(overall_left_hand_avg)[::-1][:10]
        top_right_hand = np.argsort(overall_right_hand_avg)[::-1][:10]
        
        html_content += "<h3>Top Moving Points (Indices)</h3>"
        html_content += "<p><strong>Pose:</strong> " + ", ".join(map(str, top_pose)) + "</p>"
        html_content += "<p><strong>Left Hand:</strong> " + ", ".join(map(str, top_left_hand)) + "</p>"
        html_content += "<p><strong>Right Hand:</strong> " + ", ".join(map(str, top_right_hand)) + "</p>"
    
    html_content += "</body></html>"
    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"HTML report generated at: {output_html}")

if __name__ == "__main__":
    main_dir = "../all_outputs/output_ssl_small/"  # Replace with your directory path
    output_html = "./report.html"  # Replace with your output file path
    generate_report(main_dir, output_html)