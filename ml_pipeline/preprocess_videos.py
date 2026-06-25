import os
from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────
RAW_DIR = "/Users/sandeep/Library/CloudStorage/GoogleDrive-sandeep.ghosh29@gmail.com/My Drive/Exercise_Data"
CLEAN_DIR = "/Users/sandeep/Library/CloudStorage/GoogleDrive-sandeep.ghosh29@gmail.com/My Drive/Exercise_Data_Cleaned"
MIN_VIDEOS_THRESHOLD = 30  # If a class has fewer than this, we augment it!

def standardize_video(input_path, output_path):
    """Resizes to 480p, caps duration to 5s, sets FPS to 30, and drops audio."""
    try:
        base_clip = VideoFileClip(input_path)
        
        # Cap video length at 5 seconds to prevent massive yoga files from taking over the dataset
        if base_clip.duration and base_clip.duration > 5.0:
            clip = base_clip.subclip(0, 5.0)
        else:
            clip = base_clip
            
        resized_clip = clip.resize(height=480)
        # audio=False is a HUGE performance boost because MediaPipe only needs visuals!
        resized_clip.write_videofile(output_path, fps=30, audio=False, logger=None)
        
        base_clip.close()
        if clip != base_clip:
            clip.close()
        resized_clip.close()
        return True
    except Exception as e:
        print(f"\n⚠️ Error processing {input_path}: {e}")
        return False

def create_flipped_video(input_path, output_path):
    """Creates a mirrored version of the video (Left becomes Right), capped at 5s."""
    try:
        base_clip = VideoFileClip(input_path)
        
        if base_clip.duration and base_clip.duration > 5.0:
            clip = base_clip.subclip(0, 5.0)
        else:
            clip = base_clip
            
        # Resize first to save memory before flipping
        resized_clip = clip.resize(height=480)
        flipped_clip = resized_clip.fx(vfx.mirror_x)
        flipped_clip.write_videofile(output_path, fps=30, audio=False, logger=None)
        
        base_clip.close()
        if clip != base_clip:
            clip.close()
        resized_clip.close()
        flipped_clip.close()
        return True
    except Exception as e:
        print(f"\n⚠️ Error flipping {input_path}: {e}")
        return False

def process_dataset():
    print(f"{'='*60}")
    print(f"  🎬 ENTERPRISE DATASET PREPROCESSOR")
    print(f"{'='*60}")
    
    os.makedirs(CLEAN_DIR, exist_ok=True)
    
    # Get all class folders (ignore files like .md reports or hidden folders)
    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d)) and not d.startswith("_")]
    
    for class_name in sorted(classes):
        raw_class_dir = os.path.join(RAW_DIR, class_name)
        clean_class_dir = os.path.join(CLEAN_DIR, class_name)
        os.makedirs(clean_class_dir, exist_ok=True)
        
        # Get all MP4s in this class
        videos = [f for f in os.listdir(raw_class_dir) if f.lower().endswith('.mp4')]
        video_count = len(videos)
        
        if video_count == 0:
            continue
            
        needs_augmentation = video_count < MIN_VIDEOS_THRESHOLD
        
        print(f"\nProcessing '{class_name}' ({video_count} videos)")
        if needs_augmentation:
            print(f"  -> Augmenting data (Targeting < {MIN_VIDEOS_THRESHOLD})")
            
        # Add tqdm progress bar for the videos in this class
        for vid in tqdm(videos, desc=f"  {class_name}", leave=False):
            input_path = os.path.join(raw_class_dir, vid)
            
            # 1. Standardize the original video
            standard_output = os.path.join(clean_class_dir, f"clean_{vid}")
            if not os.path.exists(standard_output):
                standardize_video(input_path, standard_output)
                
            # 2. Apply Augmentation if class is too small
            if needs_augmentation:
                flipped_output = os.path.join(clean_class_dir, f"flip_clean_{vid}")
                if not os.path.exists(flipped_output):
                    create_flipped_video(input_path, flipped_output)

    print(f"\n{'='*60}")
    print(f"✅ Preprocessing Complete! Clean dataset saved to:\n{CLEAN_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    process_dataset()