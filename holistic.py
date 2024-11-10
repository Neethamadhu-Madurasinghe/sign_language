# import cv2
# import mediapipe as mp
# import os


# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic


# # Initialize Mediapipe holistic and drawing utilities
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Specify the path to your video file
# video_path = './inputs/w00720221216120558.mp4'

# # Create the outputs directory if it doesn't exist
# output_dir = 'outputs'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Extract the file name from the video path and create the output file path
# video_filename = os.path.basename(video_path)
# output_path = os.path.join(output_dir, video_filename)

# # Capture the video
# cap = cv2.VideoCapture(video_path)

# # Get the width, height, and frames per second (fps) of the input video
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Define the codec and create a VideoWriter object to save the output video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# with mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as holistic:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty frame or end of video.")
#       break

#     # To improve performance, optionally mark the image as not writeable to pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = holistic.process(image)

#     # Draw landmark annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_CONTOURS,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp_drawing_styles
#         .get_default_face_mesh_contours_style())
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles
#         .get_default_pose_landmarks_style())
    
#     # Write the processed frame to the output video file
#     out.write(image)

#     # Display the processed video frame
#     cv2.imshow('MediaPipe Holistic', image)
#     if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
#       break

# # Release the video objects and close all windows
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"Processed video saved as: {output_path}")














import cv2
import mediapipe as mp
import os
import asyncio

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

video_path = './inputs/'
output_dir = './all_outputs/output_holistic'
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

    with mp_holistic.Holistic(
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty frame or end of video.")
                break

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.face_landmarks,
            #     mp_holistic.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        
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

