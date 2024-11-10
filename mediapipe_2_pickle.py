import cv2
import mediapipe as mp
import os
import pickle
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

num_workers = 12 
video_path = './inputs/'
output_dir = './all_outputs/output_holistic_pickle'
video_files = []

def read_all_files(path):
    """Reads all the file names in video_path directory and adds them to video_files array."""
    for filename in os.listdir(path):
        video_files.append(os.path.join(path, filename))

def extract_landmark_data(landmarks):
    """Helper function to convert Mediapipe landmark data into a list of dictionaries."""
    if landmarks:
        return [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in landmarks]
    else:
        return None

def process_video(video_path):
    """Extracts landmarks from a video and saves them to a pickle file."""
    video_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(video_filename)[0]}.pkl")

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    video_landmarks = []

    with mp_holistic.Holistic(
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print(f"End of video or empty frame: {video_path}")
                break

            # Convert the image color space from BGR to RGB for processing
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            if results.pose_landmarks:
                image_height, image_width, _ = image_rgb.shape
                image_center_x = image_width // 2
                desired_head_y = image_height // 3

                # Calculate offsets
                xs = [lm.x for lm in results.pose_landmarks.landmark]
                ys = [lm.y for lm in results.pose_landmarks.landmark]
                landmark_center_x = int(sum(xs) / len(xs) * image_width)
                head_landmark = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
                head_y = int(head_landmark.y * image_height)

                offset_x = image_center_x - landmark_center_x
                offset_y = desired_head_y - head_y
                blank_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    translated_x = int(landmark.x * image_width + offset_x)
                    translated_y = int(landmark.y * image_height + offset_y)
                    results.pose_landmarks.landmark[i].x = translated_x / image_width
                    results.pose_landmarks.landmark[i].y = translated_y / image_height
                
                mp_drawing.draw_landmarks(blank_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                if results.left_hand_landmarks:
                    for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                        translated_x = int(landmark.x * image_width + offset_x)
                        translated_y = int(landmark.y * image_height + offset_y)
                        results.left_hand_landmarks.landmark[i].x = translated_x / image_width
                        results.left_hand_landmarks.landmark[i].y = translated_y / image_height

                if results.right_hand_landmarks:
                    for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                        translated_x = int(landmark.x * image_width + offset_x)
                        translated_y = int(landmark.y * image_height + offset_y)
                        results.right_hand_landmarks.landmark[i].x = translated_x / image_width
                        results.right_hand_landmarks.landmark[i].y = translated_y / image_height

            frame_landmarks = {
                'pose_landmarks': extract_landmark_data(results.pose_landmarks.landmark) if results.pose_landmarks else None,
                'face_landmarks': extract_landmark_data(results.face_landmarks.landmark) if results.face_landmarks else None,
                'left_hand_landmarks': extract_landmark_data(results.left_hand_landmarks.landmark) if results.left_hand_landmarks else None,
                'right_hand_landmarks': extract_landmark_data(results.right_hand_landmarks.landmark) if results.right_hand_landmarks else None,
            }
            video_landmarks.append(frame_landmarks)

    cap.release()

    # Save landmarks to a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(video_landmarks, f)

    print(f"Landmarks for video saved as: {output_path}")

async def main():
    read_all_files(video_path)
    print(f"Found {len(video_files)} videos")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use ProcessPoolExecutor for parallel processing of videos
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        await asyncio.gather(*[asyncio.to_thread(executor.submit, process_video, video_file) for video_file in video_files])

    print(f"Completed processing {len(video_files)} videos")

# Run the main function
asyncio.run(main())
