import pickle

# File path
file_path = '../all_outputs/output_ssl_small/w003/2-1-w00320221110095852.mp4_1.pkl'

# Load the pickle file
def load_single_video(file_path):
    with open(file_path, 'rb') as f:
        landmarks = pickle.load(f)
    return landmarks

# Print specific pose landmarks and left hand status
def print_points(landmarks):
    points_of_interest = [15, 17, 19, 21]  # Left wrist, pinky, index, thumb
    
    for i, frame in enumerate(landmarks):
        print(f"\nFrame {i}:")
        
        # Pose landmarks
        pose_landmarks = frame['pose_landmarks']
        if pose_landmarks is not None:
            for idx in points_of_interest:
                point = pose_landmarks[idx]
                if point is not None:
                    print(f"  Pose {idx}: x={point['x']:.3f}, y={point['y']:.3f}")
                else:
                    print(f"  Pose {idx}: None")
        else:
            print("  Pose landmarks: None")
        
        # Left hand landmarks status
        left_hand_landmarks = frame['left_hand_landmarks']
        if left_hand_landmarks is not None:
            wrist = left_hand_landmarks[0]
            if wrist is not None:
                print(f"  Left hand wrist: x={wrist['x']:.3f}, y={wrist['y']:.3f}")
            else:
                print("  Left hand wrist: None")
            # Optionally print a few finger points to compare
            for idx in [4, 8]:  # Thumb tip, index tip
                point = left_hand_landmarks[idx]
                if point is not None:
                    print(f"  Left hand {idx}: x={point['x']:.3f}, y={point['y']:.3f}")
                else:
                    print(f"  Left hand {idx}: None")
        else:
            print("  Left hand landmarks: None")

# Main execution
landmarks = load_single_video(file_path)
print_points(landmarks)