import mediapipe as mp # MUST BE IMPORTED BEFORE CV2 ON MACS TO PREVENT CRASH
import cv2
import os
import csv
from tqdm import tqdm # A lifesaver for tracking progress!!

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)

CLEAN_DIR = "/Users/sandeep/Library/CloudStorage/GoogleDrive-sandeep.ghosh29@gmail.com/My Drive/Exercise_Data_Cleaned"
CSV_OUTPUT_PATH = "ml_pipeline/exercise_dataset.csv"

def init_csv():
    # ---------------------------------------------------------
    # SERIOUS BUSINESS: Setting up the 135 column Data Schema
    # ---------------------------------------------------------
    headers = ['class_name', 'video_name', 'frame_number']
    
    # MediaPipe spits out 33 landmarks. Each has an X, Y, Z, and a Visibility score.
    # 33 * 4 = 132 columns of pure math.
    for i in range(33):
        headers.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z', f'landmark_{i}_v'])
    
    with open(CSV_OUTPUT_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def process_videos():
    init_csv()

    classes = [d for d in os.listdir(CLEAN_DIR) if not d.startswith(".")]

    with open(CSV_OUTPUT_PATH, mode='a', newline='') as f:
        csv_writer = csv.writer(f)

        for class_name in sorted(classes):
            class_path = os.path.join(CLEAN_DIR, class_name)
            videos = [v for v in os.listdir(class_path) if v.endswith('.mp4')]

            print(f"\nExtracting skeletons for: {class_name}")

            for video_name in tqdm(videos, leave=False):
                video_path = os.path.join(class_path, video_name)
                cap = cv2.VideoCapture(video_path)
                frame_num = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break # Video is officially over. RIP.
                        
                    frame_num += 1
                    
                    # ---------------------------------------------------------
                    # THE BGR2RGB
                    # OpenCV reads colors backwards (Blue-Green-Red).
                    # Google MediaPipe is strictly Red-Green-Blue.
                    # If we don't flip the colors, the AI thinks the users are Smurfs.
                    # ---------------------------------------------------------
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Fixed your typo here! BGR2RB -> BGR2RGB
                    
                    # Feed the frame into the belly of the Google Neural Network
                    results = pose.process(image_rgb)
                    
                    # Did the AI actually see a human in this frame?
                    if results.pose_landmarks:
                        row = [class_name, video_name, frame_num]
                        
                        # Loop through all 33 bones and extract the raw floating point math
                        for landmark in results.pose_landmarks.landmark: # Fixed typo: pose_landmark -> pose_landmarks
                            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                            
                        # Dump it into the giant CSV spreadsheet
                        csv_writer.writerow(row)
                        
                cap.release() # Free up the RAM before your Mac explodes

if __name__ == "__main__":
    print("Starting Enterprise Feature Extraction...")
    process_videos()
    print(f"Finished! Massive dataset saved to {CSV_OUTPUT_PATH}")