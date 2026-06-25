"""
Enterprise Dataset Downloader
Downloads multiple Kaggle exercise/yoga video datasets,
merges them into a single unified folder structure,
and deduplicates overlapping classes.
"""

import os
import shutil
import kaggle
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────
BASE_DIR = "/Users/sandeep/Library/CloudStorage/GoogleDrive-sandeep.ghosh29@gmail.com/My Drive/Exercise_Data"
TEMP_DIR = os.path.join(BASE_DIR, "_temp_downloads")
REPORT_FILE = os.path.join(BASE_DIR, "dataset_report.md")

# All the Kaggle datasets we want to merge
DATASETS = [
    "hasyimabdillah/workoutfitness-video",       # ~4.6 GB, ~22 gym exercise classes
    "nandwalritik/yoga-pose-videos-dataset",      # ~1.1 GB, 6 yoga pose classes
    "pulaksarmah/yoga-videos",                    # ~936 MB, 5 yoga video classes
]

# ── DEDUPLICATION MAP ───────────────────────────────────────────────
# If two datasets have folders that mean the same exercise but are
# named differently, this map merges them into ONE canonical name.
# Format: "original_folder_name_lowercase" -> "canonical_name"
# We will normalize everything to lowercase with underscores.
MERGE_MAP = {
    # Squats variants
    "squat": "squats",
    "bodyweight squat": "squats",
    "bodyweight squats": "squats",
    "barbell squat": "squats",
    "goblet squat": "squats",

    # Pushup variants
    "push up": "pushups",
    "push-up": "pushups",
    "push ups": "pushups",
    "push-ups": "pushups",

    # Pullup variants
    "pull up": "pullups",
    "pull-up": "pullups",
    "pull ups": "pullups",
    "pull-ups": "pullups",
    "chin up": "pullups",
    "chin-up": "pullups",

    # Deadlift variants
    "deadlift": "deadlifts",
    "dead lift": "deadlifts",

    # Bench press variants
    "bench press": "bench_press",
    "benchpress": "bench_press",

    # Bicep curl variants
    "bicep curl": "bicep_curls",
    "biceps curl": "bicep_curls",
    "barbell biceps curl": "bicep_curls",
    "dumbbell bicep curl": "bicep_curls",
    "dumbbell biceps curl": "bicep_curls",

    # Shoulder press variants
    "shoulder press": "shoulder_press",
    "overhead press": "shoulder_press",
    "military press": "shoulder_press",

    # Yoga pose name standardization (Sanskrit -> English)
    "bhujangasana": "cobra_pose",
    "padmasana": "lotus_pose",
    "shavasana": "corpse_pose",
    "tadasana": "mountain_pose",
    "trikonasana": "triangle_pose",
    "vrikshasana": "tree_pose",
}


def normalize_name(folder_name):
    """
    Takes a raw folder name from any dataset and returns
    a clean, canonical class name.
    """
    # Lowercase and strip whitespace
    clean = folder_name.strip().lower()
    # Replace hyphens and spaces with underscores
    clean = clean.replace("-", " ").replace("_", " ")

    # Check if this name has a canonical mapping
    if clean in MERGE_MAP:
        return MERGE_MAP[clean]

    # Otherwise just convert spaces to underscores
    return clean.replace(" ", "_")


def download_and_extract(dataset_slug):
    """Downloads a Kaggle dataset into a temporary folder and unzips it."""
    dest = os.path.join(TEMP_DIR, dataset_slug.replace("/", "__"))
    os.makedirs(dest, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Downloading: {dataset_slug}")
    print(f"  Into: {dest}")
    print(f"{'='*60}")
    kaggle.api.dataset_download_files(dataset_slug, path=dest, unzip=True)
    return dest


def find_video_folders(root_path):
    """
    Walks through the unzipped dataset and finds all folders
    that contain .mp4 files (those are the exercise class folders).
    Returns a list of (folder_path, folder_name) tuples.
    """
    video_folders = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        mp4_files = [f for f in filenames if f.lower().endswith(".mp4")]
        if mp4_files:
            folder_name = os.path.basename(dirpath)
            video_folders.append((dirpath, folder_name, len(mp4_files)))
    return video_folders


def merge_into_unified(video_folders):
    """
    Takes all discovered video folders and copies their .mp4 files
    into the unified BASE_DIR structure using canonical names.
    """
    stats = {}
    for folder_path, raw_name, count in video_folders:
        canonical = normalize_name(raw_name)

        # Skip temp/junk folders
        if canonical.startswith("_") or canonical in ("temp", "tmp"):
            continue

        dest_folder = os.path.join(BASE_DIR, canonical)
        os.makedirs(dest_folder, exist_ok=True)

        # Collect all mp4 files first
        mp4_files = []
        for f in os.listdir(folder_path):
            if f.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(folder_path, f))

        # Now process them with a progress bar
        copied = 0
        for file_path in tqdm(mp4_files, desc=f"Processing {raw_name}", leave=False):
            dst = os.path.join(dest_folder, os.path.basename(file_path))
            # Don't overwrite if a file with the same name already exists
            if not os.path.exists(dst):
                shutil.copy2(file_path, dst)
                copied += 1

        if canonical not in stats:
            stats[canonical] = {"count": 0, "size": 0}
        stats[canonical]["count"] += copied
        # Calculate size of copied files
        for f in os.listdir(dest_folder):
            stats[canonical]["size"] += os.path.getsize(os.path.join(dest_folder, f))
        
        print(f"  ✅ '{raw_name}' -> '{canonical}' ({copied} videos copied)")

    return stats


def cleanup_temp():
    """Removes the temporary download folder."""
    if os.path.exists(TEMP_DIR):
        print(f"\n🧹 Cleaning up temporary files...")
        shutil.rmtree(TEMP_DIR)
        print("   Done!")


# ── MAIN ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    # Step 1: Download all datasets
    extracted_paths = []
    for slug in DATASETS:
        try:
            path = download_and_extract(slug)
            extracted_paths.append(path)
        except Exception as e:
            print(f"  ❌ Failed to download {slug}: {e}")
            continue

    # Step 2: Find all folders containing video files
    print(f"\n{'='*60}")
    print("  Scanning for exercise video folders...")
    print(f"{'='*60}")
    all_video_folders = []
    for path in extracted_paths:
        folders = find_video_folders(path)
        all_video_folders.extend(folders)
        print(f"  Found {len(folders)} classes in {path}")

    # Step 3: Merge everything into the unified structure
    print(f"\n{'='*60}")
    print("  Merging into unified dataset...")
    print(f"{'='*60}")
    stats = merge_into_unified(all_video_folders)

    # Step 4: Cleanup temp downloads
    cleanup_temp()

    # Step 5: Print and save the final report
    print(f"\n{'='*60}")
    print(f"  🏋️ FINAL DATASET REPORT")
    print(f"{'='*60}")
    total_videos = 0
    total_size_bytes = 0
    
    report_content = "# 🏋️ Master Exercise Dataset Report\n\n"
    report_content += "This report contains the details of all the unified exercise classes downloaded from Kaggle.\n\n"
    report_content += "| Class Name | Video Count | Size (MB) |\n"
    report_content += "|---|---|---|\n"

    for class_name in sorted(stats.keys()):
        count = stats[class_name]["count"]
        size_bytes = stats[class_name]["size"]
        size_mb = size_bytes / (1024 * 1024)
        
        total_videos += count
        total_size_bytes += size_bytes
        
        print(f"  {class_name:.<40} {count:>4} videos ({size_mb:.2f} MB)")
        report_content += f"| {class_name} | {count} | {size_mb:.2f} MB |\n"

    total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
    
    print(f"{'='*60}")
    print(f"  Total Classes: {len(stats)}")
    print(f"  Total Videos:  {total_videos}")
    print(f"  Total Size:    {total_size_gb:.2f} GB")
    print(f"  Location:      {BASE_DIR}")
    print(f"{'='*60}")
    
    report_content += "\n## Summary\n"
    report_content += f"- **Total Classes**: {len(stats)}\n"
    report_content += f"- **Total Videos**: {total_videos}\n"
    report_content += f"- **Total Size**: {total_size_gb:.2f} GB\n"
    report_content += f"- **Location**: `{BASE_DIR}`\n"
    
    with open(REPORT_FILE, "w") as f:
        f.write(report_content)
        
    print(f"\n📄 A detailed markdown report has been saved to: {REPORT_FILE}")