#!/bin/bash
set -e

echo "=========================================================="
echo "🎯 EXTRACTING LANDMARKS (via MediaPipe mp_env)..."
echo "=========================================================="
for vid in test_videos/*_short.mp4; do
    base=$(basename "$vid" .mp4)
    echo "Processing $vid -> test_videos/${base}_landmarks.csv"
    mp_env/bin/python ml_pipeline/extract_video_landmarks.py "$vid" "test_videos/${base}_landmarks.csv"
done

echo "=========================================================="
echo "🤖 EVALUATING PIPELINE (via TensorFlow)..."
echo "=========================================================="
for csv in test_videos/*_short_landmarks.csv; do
    echo "----------------------------------------------------------"
    echo "Testing $csv"
    python3.10 ml_pipeline/evaluate_custom_csv.py "$csv"
done
