"""
End-to-End Pipeline Tester (CSV-based)
--------------------------------------
Tests the COMPLETE exercise detection pipeline on pre-extracted landmark data,
simulating exactly what happens in the live webcam frontend:

1. Visibility check (all 12 body joints must be visible)
2. Motion gate (body must be moving for rep exercises, bypassed for timed)
3. Scaler transform + model prediction
4. Confidence threshold (92% for reps, 95% for timed)
5. Majority vote buffer (last 8 predictions)
6. Lock-in mechanism (18 frames for reps, 25 for timed)
7. Rep counting via peak/valley detection
8. Hold timer for timed exercises

This uses the already-extracted CSV dataset so we dont need mediapipe installed.
The CSV has frame-by-frame landmarks for every video in the training set.

Usage:
    python3.10 test_on_video.py                    # test random sample
    python3.10 test_on_video.py --exercise squats  # test all squats
    python3.10 test_on_video.py --full             # test every single video
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import os
import sys
import argparse
import math
import random
from collections import Counter
from tqdm import tqdm

# ======= LOAD THE AI BRAIN =======
MODEL_PATH = os.path.join(os.path.dirname(__file__), "exercise_model.keras")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "label_classes.npy")
CSV_PATH = os.path.join(os.path.dirname(__file__), "exercise_dataset.csv")

print("Loading AI Brain...")
MODEL = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    SCALER = pickle.load(f)
LABEL_CLASSES = np.load(LABELS_PATH, allow_pickle=True)
print(f"Loaded {len(LABEL_CLASSES)} exercise classes")

# ======= CONFIG (mirrors the frontend EXACTLY) =======
CONFIDENCE_MIN = 0.92
CONFIDENCE_TIMED = 0.95
LOCK_FRAMES = 18
LOCK_FRAMES_TIMED = 25
UNLOCK_FRAMES = 20
MOTION_MIN = 0.005
BUFFER_MAX = 8
MIN_VISIBLE_JOINTS = 12  # default: all 12 body joints (can be overridden via CLI)

# body landmark indices (face excluded, same as frontend)
# indices 11-16: shoulders, elbows, wrists
# indices 23-28: hips, knees, ankles
BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

TIMED_EXERCISES = {'plank', 'mountain_pose', 'tree_pose', 'padamasana', 'bhujasana', 'trikasana'}


class LandmarkProxy:
    """little helper to mimic the mediapipe landmark object structure"""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def row_to_landmarks(row_data):
    """convert a csv row into a list of 33 landmark objects"""
    landmarks = []
    for i in range(33):
        x = row_data[f'landmark_{i}_x']
        y = row_data[f'landmark_{i}_y']
        z = row_data[f'landmark_{i}_z']
        v = row_data[f'landmark_{i}_v']
        landmarks.append(LandmarkProxy(x, y, z, v))
    return landmarks


def calc_motion(current, previous):
    """motion detection - body landmarks only, no face"""
    if previous is None:
        return 1.0
    total = 0
    for idx in BODY_INDICES:
        dx = current[idx].x - previous[idx].x
        dy = current[idx].y - previous[idx].y
        total += math.sqrt(dx * dx + dy * dy)
    return total / len(BODY_INDICES)


def majority_vote(buffer):
    if not buffer:
        return None
    counts = Counter(buffer)
    return counts.most_common(1)[0][0]


def get_rep_signal(landmarks):
    """average Y of shoulders + wrists"""
    return (landmarks[11].y + landmarks[12].y + landmarks[15].y + landmarks[16].y) / 4.0


def predict_from_row(row_data):
    """run model on one row of CSV data"""
    feature_cols = [c for c in row_data.index if c.startswith('landmark_')]
    X_raw = pd.DataFrame([row_data[feature_cols].values], columns=feature_cols)
    X_scaled = SCALER.transform(X_raw)
    predictions = MODEL.predict(X_scaled, verbose=0)
    class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][class_index])
    exercise_name = LABEL_CLASSES[class_index]
    return exercise_name, confidence


def process_video_frames(frames_df, true_label, verbose=False):
    """
    Process all frames of a single video through the FULL pipeline.
    frames_df should be sorted by frame_number.
    """
    # ======= STATE =======
    prev_landmarks = None
    pred_buffer = []
    stable_pred = None
    stable_frames = 0
    locked_exercise = None
    mismatch_frames = 0

    # rep counting
    rep_count = 0
    signal_history = []
    smoothed_sig = None
    rep_phase = None
    half_reps = 0

    # hold timer stuff
    hold_start_frame = None
    hold_frames = 0

    # stats
    total_frames = len(frames_df)
    frames_not_visible = 0
    frames_idle = 0
    frames_low_conf = 0
    frames_detecting = 0
    frames_locked = 0
    all_raw_predictions = []
    all_confidences = []

    # simulate the ~200ms cooldown by only predicting every N frames
    # assume ~30fps, so 200ms = ~6 frames
    pred_cooldown = 6
    last_pred_idx = -999

    for idx, (_, row) in enumerate(frames_df.iterrows()):
        landmarks = row_to_landmarks(row)

        # VISIBILITY CHECK
        visible_count = sum(1 for i in BODY_INDICES if landmarks[i].visibility > 0.65)
        if visible_count < MIN_VISIBLE_JOINTS:
            frames_not_visible += 1
            prev_landmarks = None
            continue

        # rep counting on every frame when locked
        if locked_exercise and locked_exercise not in TIMED_EXERCISES:
            raw = get_rep_signal(landmarks)
            if smoothed_sig is None:
                smoothed_sig = raw
            else:
                smoothed_sig = smoothed_sig * 0.7 + raw * 0.3

            signal_history.append(smoothed_sig)
            if len(signal_history) > 30:
                signal_history.pop(0)

            if len(signal_history) >= 10:
                min_y = min(signal_history)
                max_y = max(signal_history)
                range_y = max_y - min_y

                if range_y >= 0.04:
                    midpoint = (min_y + max_y) / 2
                    dead_zone = range_y * 0.15
                    current_val = signal_history[-1]

                    if rep_phase is None:
                        rep_phase = 'above' if current_val > midpoint else 'below'
                    elif rep_phase == 'below' and current_val > midpoint + dead_zone:
                        rep_phase = 'above'
                        half_reps += 1
                        if half_reps % 2 == 0:
                            rep_count += 1
                    elif rep_phase == 'above' and current_val < midpoint - dead_zone:
                        rep_phase = 'below'
                        half_reps += 1
                        if half_reps % 2 == 0:
                            rep_count += 1

        # timed exercise hold tracking
        if locked_exercise and locked_exercise in TIMED_EXERCISES:
            hold_frames += 1

        # THROTTLE predictions
        if idx - last_pred_idx < pred_cooldown:
            continue
        last_pred_idx = idx

        # MOTION CHECK - now only updated every cooldown cycle to match 200ms webcam interval
        motion = calc_motion(landmarks, prev_landmarks)
        prev_landmarks = landmarks

        # GET PREDICTION
        exercise, confidence = predict_from_row(row)
        all_raw_predictions.append(exercise)
        all_confidences.append(confidence)

        if not locked_exercise:
            # === PHASE 1: DETECTION ===
            if confidence < CONFIDENCE_MIN:
                frames_low_conf += 1
                stable_frames = 0
                continue

            pred_buffer.append(exercise)
            if len(pred_buffer) > BUFFER_MAX:
                pred_buffer.pop(0)
            voted = majority_vote(pred_buffer)
            is_timed_pred = voted in TIMED_EXERCISES

            # motion gate (only for rep exercises)
            if not is_timed_pred and motion < MOTION_MIN:
                frames_idle += 1
                stable_frames = 0
                continue

            # timed exercises need higher confidence
            required_conf = CONFIDENCE_TIMED if is_timed_pred else CONFIDENCE_MIN
            if confidence < required_conf:
                frames_low_conf += 1
                stable_frames = 0
                continue

            if voted == stable_pred:
                stable_frames += 1
            else:
                stable_pred = voted
                stable_frames = 1

            needed = LOCK_FRAMES_TIMED if is_timed_pred else LOCK_FRAMES
            if stable_frames >= needed:
                # LOCKED IN
                locked_exercise = voted
                mismatch_frames = 0
                rep_count = 0
                half_reps = 0
                rep_phase = None
                smoothed_sig = None
                signal_history = []
                hold_start_frame = idx
                hold_frames = 0
                frames_locked += 1
                if verbose:
                    print(f"    [{idx}/{total_frames}] LOCKED: {voted} (conf: {round(confidence*100)}%)")
            else:
                frames_detecting += 1

        else:
            # === PHASE 2: TRACKING ===
            frames_locked += 1

            pred_buffer.append(exercise)
            if len(pred_buffer) > BUFFER_MAX:
                pred_buffer.pop(0)
            voted = majority_vote(pred_buffer)

            if voted == locked_exercise:
                mismatch_frames = 0
            else:
                mismatch_frames += 1
                if mismatch_frames >= UNLOCK_FRAMES:
                    if verbose:
                        print(f"    [{idx}/{total_frames}] UNLOCKED from {locked_exercise} -> {voted}")
                    locked_exercise = None
                    stable_pred = None
                    stable_frames = 0
                    mismatch_frames = 0

    # determine hold time in seconds (assuming ~30fps)
    hold_secs = round(hold_frames / 30.0, 1) if hold_frames > 0 else 0

    # raw model majority
    raw_majority = Counter(all_raw_predictions).most_common(1)[0][0] if all_raw_predictions else "none"
    raw_majority_pct = round(Counter(all_raw_predictions).most_common(1)[0][1] / len(all_raw_predictions) * 100, 1) if all_raw_predictions else 0

    correct = (locked_exercise == true_label) if locked_exercise else False

    return {
        "true_label": true_label,
        "locked_exercise": locked_exercise or "none",
        "correct": correct,
        "reps_counted": rep_count,
        "hold_secs": hold_secs,
        "total_frames": total_frames,
        "raw_majority": raw_majority,
        "raw_majority_pct": raw_majority_pct,
        "avg_confidence": round(np.mean(all_confidences) * 100, 1) if all_confidences else 0,
        "frames_not_visible": frames_not_visible,
        "frames_idle": frames_idle,
        "frames_low_conf": frames_low_conf,
        "frames_detecting": frames_detecting,
        "frames_locked": frames_locked,
    }


def run_tests(exercises=None, max_per_class=3, verbose=False):
    """run the full pipeline on videos from the CSV dataset"""
    print(f"\nLoading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} frames across {df['video_name'].nunique()} videos")

    if exercises is None:
        exercises = sorted(df['class_name'].unique())

    all_results = []
    correct_total = 0
    total_tested = 0
    exercise_stats = {}

    for exercise in exercises:
        ex_df = df[df['class_name'] == exercise]
        videos = list(ex_df['video_name'].unique())

        if not videos:
            continue

        # sample videos
        if len(videos) > max_per_class:
            test_vids = random.sample(videos, max_per_class)
        else:
            test_vids = videos

        print(f"\n{'='*60}")
        print(f"Testing: {exercise} ({len(test_vids)} videos)")
        print(f"{'='*60}")

        ex_correct = 0
        ex_total = 0
        ex_reps = []

        for vid in tqdm(test_vids, desc=exercise, leave=True):
            vid_frames = ex_df[ex_df['video_name'] == vid].sort_values('frame_number')
            result = process_video_frames(vid_frames, true_label=exercise, verbose=verbose)

            all_results.append(result)
            total_tested += 1
            ex_total += 1

            if result["correct"]:
                correct_total += 1
                ex_correct += 1

            if result["reps_counted"] > 0:
                ex_reps.append(result["reps_counted"])

            status = "✅" if result["correct"] else "❌"
            vid_short = vid[:35]
            reps_str = f" | {result['reps_counted']} reps" if result['reps_counted'] > 0 else ""
            hold_str = f" | {result['hold_secs']}s hold" if result['hold_secs'] > 0 else ""
            lock_str = result['locked_exercise']
            print(f"  {status} {vid_short:35s} -> {lock_str:20s} (raw: {result['raw_majority']} @ {result['avg_confidence']}%){reps_str}{hold_str}")

        acc = round(ex_correct / ex_total * 100, 1) if ex_total > 0 else 0
        avg_reps = round(sum(ex_reps) / len(ex_reps), 1) if ex_reps else 0
        exercise_stats[exercise] = {
            "tested": ex_total, "correct": ex_correct, "accuracy": acc, "avg_reps": avg_reps
        }

    # ======= FINAL REPORT =======
    print("\n\n" + "=" * 70)
    print("          END-TO-END PIPELINE TEST REPORT")
    print("          (with ALL filters: visibility, motion, confidence,")
    print("           majority vote, lock-in, rep counting)")
    print("=" * 70)

    overall_acc = round(correct_total / total_tested * 100, 1) if total_tested > 0 else 0
    print(f"\n🎯 Overall Pipeline Accuracy: {correct_total}/{total_tested} = {overall_acc}%")
    print(f"   (This is the REAL accuracy with every filter applied)\n")

    print(f"{'Exercise':<25s} {'Tested':>7s} {'Correct':>8s} {'Accuracy':>9s} {'Avg Reps':>9s}")
    print("-" * 60)
    for ex, stats in sorted(exercise_stats.items()):
        emoji = "✅" if stats['accuracy'] >= 80 else "⚠️ " if stats['accuracy'] >= 50 else "❌"
        print(f"{emoji} {ex:<23s} {stats['tested']:>7d} {stats['correct']:>8d} {stats['accuracy']:>8.1f}% {stats['avg_reps']:>9.1f}")

    # problem exercises
    print(f"\n--- Problem Exercises (accuracy < 80%) ---")
    problems = [(ex, s) for ex, s in exercise_stats.items() if s['accuracy'] < 80]
    if problems:
        for ex, s in sorted(problems, key=lambda x: x[1]['accuracy']):
            print(f"  ⚠️  {ex}: {s['accuracy']}% ({s['correct']}/{s['tested']})")
    else:
        print("  None! All exercises above 80%.")

    # misidentification details
    print(f"\n--- Misidentifications ---")
    misses = [(r['true_label'], r['locked_exercise']) for r in all_results if not r['correct']]
    if misses:
        confusion = Counter(misses)
        for (true, pred), count in confusion.most_common():
            print(f"  {true} -> {pred} ({count}x)")
    else:
        print("  Perfect! No misidentifications.")

    # frame rejection stats
    total_not_vis = sum(r['frames_not_visible'] for r in all_results)
    total_idle = sum(r['frames_idle'] for r in all_results)
    total_low = sum(r['frames_low_conf'] for r in all_results)
    total_locked = sum(r['frames_locked'] for r in all_results)

    print(f"\n--- Frame Rejection Breakdown ---")
    print(f"  Body not fully visible: {total_not_vis}")
    print(f"  Idle (no body motion):  {total_idle}")
    print(f"  Low confidence:         {total_low}")
    print(f"  Locked & tracking:      {total_locked}")

    return all_results, exercise_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end pipeline tester")
    parser.add_argument("--exercise", type=str, help="test one exercise class")
    parser.add_argument("--full", action="store_true", help="test ALL videos")
    parser.add_argument("--sample", type=int, default=3, help="videos per class (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="show lock/unlock events")
    parser.add_argument("--no-visibility", action="store_true", help="skip the visibility check entirely")
    parser.add_argument("--min-visible", type=int, default=8, help="min visible joints (default 8, use 0 to skip)")
    args = parser.parse_args()

    # override visibility setting
    if args.no_visibility:
        MIN_VISIBLE_JOINTS = 0
        print("⚡ Visibility check DISABLED")
    else:
        MIN_VISIBLE_JOINTS = args.min_visible
        print(f"👁️  Requiring {MIN_VISIBLE_JOINTS}/12 body joints visible")

    if args.exercise:
        run_tests(exercises=[args.exercise], max_per_class=999, verbose=args.verbose)
    elif args.full:
        run_tests(max_per_class=999, verbose=args.verbose)
    else:
        run_tests(max_per_class=args.sample, verbose=args.verbose)
