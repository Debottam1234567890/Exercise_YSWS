#!/bin/bash

mkdir -p test_videos

echo "📥 Downloading a pushups tutorial..."
yt-dlp "ytsearch1:pushups perfect form" \
    -f "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
    -o "test_videos/pushups.%(ext)s" \
    --max-downloads 1 || true

echo "📥 Downloading a squats tutorial..."
yt-dlp "ytsearch1:squats perfect form" \
    -f "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
    -o "test_videos/squats.%(ext)s" \
    --max-downloads 1 || true

echo "=========================================================="
echo "🎯 EXTRACTING LANDMARKS (via MediaPipe mp_env)..."
echo "=========================================================="
for vid in test_videos/*.mp4; do
    base=$(basename "$vid" .mp4)
    echo "Processing $vid -> test_videos/${base}_landmarks.csv"
    mp_env/bin/python ml_pipeline/extract_video_landmarks.py "$vid" "test_videos/${base}_landmarks.csv"
done

echo "=========================================================="
echo "🤖 EVALUATING PIPELINE (via TensorFlow)..."
echo "=========================================================="
for csv in test_videos/*_landmarks.csv; do
    echo "----------------------------------------------------------"
    echo "Testing $csv"
    python3.10 ml_pipeline/evaluate_custom_csv.py "$csv"
done
