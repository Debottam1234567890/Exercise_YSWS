import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import os
import sys

# ======= CONFIG =======
CONFIDENCE_MIN = 0.92
CONFIDENCE_TIMED = 0.95
LOCK_FRAMES = 18
LOCK_FRAMES_TIMED = 25
UNLOCK_FRAMES = 20
MOTION_MIN = 0.005
BUFFER_MAX = 8

BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
TIMED_EXERCISES = {'plank', 'mountain_pose', 'tree_pose', 'padamasana', 'bhujasana', 'trikasana'}

def evaluate_csv(csv_path):
    print("Loading AI Brain...")
    model_path = os.path.join(os.path.dirname(__file__), "exercise_model.keras")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    labels_path = os.path.join(os.path.dirname(__file__), "label_classes.npy")
    
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    label_classes = np.load(labels_path, allow_pickle=True)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} frames from {csv_path}")
    
    import test_on_video as tv
    # Override tv configuration
    tv.MODEL = model
    tv.SCALER = scaler
    tv.LABEL_CLASSES = label_classes
    tv.MIN_VISIBLE_JOINTS = 8
    
    print("\n--- Pipeline Evaluation Timeline ---")
    result = tv.process_video_frames(df, true_label="unknown", verbose=True)
    
    print("\n--- Final Summary ---")
    print(f"Locked Exercise: {result['locked_exercise']}")
    if result['reps_counted'] > 0:
        print(f"Reps Counted: {result['reps_counted']}")
    if result['hold_secs'] > 0:
        print(f"Hold Time: {result['hold_secs']}s")
    
    print(f"\nStats:")
    print(f"  Frames Idle: {result['frames_idle']}")
    print(f"  Frames Low Conf: {result['frames_low_conf']}")
    print(f"  Avg Confidence: {result['avg_confidence']}%")
    print(f"  Raw Majority Pred: {result['raw_majority']} ({result['raw_majority_pct']}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate_custom_csv.py <path_to_landmarks.csv>")
        sys.exit(1)
    evaluate_csv(sys.argv[1])
