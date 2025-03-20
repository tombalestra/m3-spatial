import cv2
import os

def extract_frames(video_path, output_folder, desired_fps):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / desired_fps)
    frame_count = 0
    extracted_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{extracted_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_frame_count} frames to {output_folder} at {desired_fps} FPS")

# Usage
video_path = '/home/xueyan/code/data/nav_video/IMG_4674.MOV'
output_folder = '/home/xueyan/code/data/tandt/lab2/input'
desired_fps = 5  # Adjust this value as needed
extract_frames(video_path, output_folder, desired_fps)