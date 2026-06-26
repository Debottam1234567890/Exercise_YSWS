import cv2
import mediapipe as mp
import pandas as pd
import sys
import os

def extract_landmarks(video_path, output_csv):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None or pd.isna(fps):
        fps = 30.0

    frames_data = []
    frame_number = 0
    
    print(f"Extracting landmarks from {video_path} (FPS: {fps})...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            row = {
                'video_name': os.path.basename(video_path),
                'frame_number': frame_number,
                'class_name': 'unknown'
            }
            
            for i, lm in enumerate(results.pose_landmarks.landmark):
                row[f'landmark_{i}_x'] = lm.x
                row[f'landmark_{i}_y'] = lm.y
                row[f'landmark_{i}_z'] = lm.z
                row[f'landmark_{i}_v'] = lm.visibility
                
            frames_data.append(row)
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames...")

    cap.release()
    
    if frames_data:
        df = pd.DataFrame(frames_data)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(frames_data)} frames of landmarks to {output_csv}")
    else:
        print("No landmarks detected in the entire video.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 extract_video_landmarks.py <input_video> <output_csv>")
        sys.exit(1)
        
    extract_landmarks(sys.argv[1], sys.argv[2])
