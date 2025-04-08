import cv2
import mediapipe as mp
import pickle
import os
import numpy as np
import asyncio

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Path to the pickle file
# pickle_file_path = '../all_outputs/output_include/Adjectives/bad'  
pickle_file_path = '../all_outputs/output_ssl_small/Exam'  
output_dir = '../all_outputs/output_holistic'

pickle_files = []

def read_all_files(path):
    # Read all the file names in video_path directory and add them to video_files array
    for filename in os.listdir(path):
        pickle_files.append(os.path.join(path, filename))


def draw_landmarks_on_image(image, frame_landmarks):
    """Helper function to draw landmarks on the image."""
    if frame_landmarks['pose_landmarks']:
        pose_landmarks_list = frame_landmarks['pose_landmarks']
        for connection in mp_holistic.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(pose_landmarks_list) and end_idx < len(pose_landmarks_list):
                start_point = (int(pose_landmarks_list[start_idx]['x'] * image.shape[1]), int(pose_landmarks_list[start_idx]['y'] * image.shape[0]))
                end_point = (int(pose_landmarks_list[end_idx]['x'] * image.shape[1]), int(pose_landmarks_list[end_idx]['y'] * image.shape[0]))
                cv2.line(image, start_point, end_point, (0, 255, 0), 4)

    if frame_landmarks['face_landmarks']:
        for lm in frame_landmarks['face_landmarks']:
            cv2.circle(image, (int(lm['x'] * image.shape[1]), int(lm['y'] * image.shape[0])), 1, (255, 0, 0), -1)

    if frame_landmarks['left_hand_landmarks']:
        hand_landmarks_list = frame_landmarks['left_hand_landmarks']
        for connection in mp_holistic.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(hand_landmarks_list) and end_idx < len(hand_landmarks_list):
                start_point = (int(hand_landmarks_list[start_idx]['x'] * image.shape[1]), int(hand_landmarks_list[start_idx]['y'] * image.shape[0]))
                end_point = (int(hand_landmarks_list[end_idx]['x'] * image.shape[1]), int(hand_landmarks_list[end_idx]['y'] * image.shape[0]))
                cv2.line(image, start_point, end_point, (0, 0, 255), 4)


        for lm in frame_landmarks['left_hand_landmarks']:
            cv2.circle(image, (int(lm['x'] * image.shape[1]), int(lm['y'] * image.shape[0])), 4, (0, 0, 255), -1)

    if frame_landmarks['right_hand_landmarks']:
        hand_landmarks_list = frame_landmarks['right_hand_landmarks']
        for connection in mp_holistic.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(hand_landmarks_list) and end_idx < len(hand_landmarks_list):
                start_point = (int(hand_landmarks_list[start_idx]['x'] * image.shape[1]), int(hand_landmarks_list[start_idx]['y'] * image.shape[0]))
                end_point = (int(hand_landmarks_list[end_idx]['x'] * image.shape[1]), int(hand_landmarks_list[end_idx]['y'] * image.shape[0]))
                cv2.line(image, start_point, end_point, (255, 0, 255), 4)


        for lm in frame_landmarks['right_hand_landmarks']:
            cv2.circle(image, (int(lm['x'] * image.shape[1]), int(lm['y'] * image.shape[0])), 4, (0, 0, 255), -1)        


def read_pickle_and_draw_skeleton(pickle_file, output_dir):
    # Load the landmarks data from the pickle file
    with open(pickle_file, 'rb') as f:
        video_landmarks = pickle.load(f)
        print("number of frames")
        print(len(video_landmarks))
    
    # Define the output video path and create the directory if needed
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'skeleton_output.avi')

    # Set up video writer
    frame_height = 720
    frame_width = 1280
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'mp4v' for .mp4 files
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Create a window to display the output
    cv2.namedWindow('Skeleton', cv2.WINDOW_NORMAL)

    for i, frame_landmarks in enumerate(video_landmarks):
        # Create a blank white image
        image = 255 * np.ones(shape=[frame_height, frame_width, 3], dtype=np.uint8)

        # Draw the landmarks on the image
        draw_landmarks_on_image(image, frame_landmarks)

        # Show the image
        cv2.imshow('Skeleton', image)

        # Write the frame to the video file
        # out.write(image)

        # Wait for 50ms (20 fps) or exit if 'q' is pressed
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # Release everything when done
    out.release()
    cv2.destroyAllWindows()



async def main():
    read_all_files(pickle_file_path)

    print(f"Found {len(pickle_files)} pickle files")

    # Create the outputs directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Process each video one by one
    for pickle_file in pickle_files[:1]:
        print(f"Processing file: {pickle_file}")
        read_pickle_and_draw_skeleton(pickle_file, output_dir)
        
    print(f"Completed processing {len(pickle_file)} videos")

print("Starting main function")
asyncio.run(main())
# main()
