import cv2
import mediapipe as mp
import numpy as np
import os
import asyncio

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

video_path = './inputs/'
output_dir = './all_outputs/outputs'
video_files = []

def read_all_files(path):
    # Read all the file names in video_path directory and add them to video_files array with video_path prepend to file name eg: inputs/video.mp4
    for filename in os.listdir(path):
        video_files.append(os.path.join(path, filename))

async def detect(video_path):
    # Extract the file name from the video path and create the output file path
    video_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_filename)

    # Capture the video
    cap = cv2.VideoCapture(video_path)

    # Get the width, height, and frames per second (fps) of the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty frame or end of video.")
                break

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                
            out.write(image)
            
            # Optionally, show the video (for debugging purposes)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Processed video saved as: {output_path}")

async def main():
    read_all_files(video_path)

    print("Found " + str(len(video_files)) + " videos")

    # Create the outputs directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Process each video one by one
    for video_file in video_files:
        print("Converting video: " + video_file)
        await detect(video_file)
        
    print("Completed all " + str(len(video_files)) + " videos")

asyncio.run(main())
