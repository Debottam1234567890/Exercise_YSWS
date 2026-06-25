import csv
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms & Shoulders
    (11, 23), (12, 24), (23, 24),                     # Torso
    (23, 25), (25, 27), (24, 26), (26, 28),           # Legs
    (27, 29), (29, 31), (31, 27), (28, 30), (30, 32), (32, 28), # Feet
    (15, 17), (15, 19), (15, 21), (17, 19),           # Left Hand
    (16, 18), (16, 20), (16, 22), (18, 20),           # Right Hand
    (0, 1), (1, 2), (2, 3), (3, 7),                   # Face Left
    (0, 4), (4, 5), (5, 6), (6, 8),                   # Face Right
    (9, 10)                                           # Mouth
]

def load_data():
    print("Loading 305MB CSV into RAM (this takes ~3 seconds)...")
    videos_map = {}
    with open("ml_pipeline/exercise_dataset.csv", "r") as f:
        reader = csv.reader(f)
        next(reader) # Skip headers
        for row in reader:
            video_name = row[1]
            if video_name not in videos_map:
                videos_map[video_name] = []
            videos_map[video_name].append(row)
            
    # Sort the frames numerically inside each video so it plays in chronological order
    for vid in videos_map:
        videos_map[vid].sort(key=lambda x: int(x[2]))
        
    return list(videos_map.values())

def launch_visualizer():
    all_videos = load_data()
    if not all_videos:
        print("No data found!")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    
    # Store the currently playing video in a state dictionary
    state = {'current_video_frames': random.choice(all_videos)}

    # This function redraws a single frame of the animation
    def draw_frame(frame_index):
        ax.clear()
        ax.set_facecolor('black')
        ax.axis('off')
        
        # IMPORTANT: Lock the camera limits so the skeleton doesn't jitter rapidly!
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([-1.0, 1.0])  
        ax.set_zlim([-2.5, 0.5])  

        chosen_row = state['current_video_frames'][frame_index]
        class_name = chosen_row[0].upper()
        video_name = chosen_row[1]
        frame_num = chosen_row[2]

        ax.set_title(f"Class: {class_name}\nVideo: {video_name} (Frame: {frame_num})", color='white', fontsize=14)

        math_data = chosen_row[3:]
        x_vals, y_vals, z_vals = [], [], []
        for i in range(33):
            x_vals.append(float(math_data[i*4]))
            y_vals.append(-float(math_data[i*4 + 1])) 
            z_vals.append(-float(math_data[i*4 + 2]))

        ax.scatter(x_vals, z_vals, y_vals, c='red', s=50)

        for start_idx, end_idx in CONNECTIONS:
            ax.plot(
                [x_vals[start_idx], x_vals[end_idx]],
                [z_vals[start_idx], z_vals[end_idx]],
                [y_vals[start_idx], y_vals[end_idx]],
                c='cyan', linewidth=2
            )

    # Pick a completely new video when the button is clicked
    def pick_new_video(event=None):
        state['current_video_frames'] = random.choice(all_videos)

    # The endless animation loop
    def animate(i):
        num_frames = len(state['current_video_frames'])
        real_frame = i % num_frames # modulo operator loops the video seamlessly!
        draw_frame(real_frame)
        
    # Set interval to 33ms (~30 FPS playback)
    ani = animation.FuncAnimation(fig, animate, interval=33, cache_frame_data=False)

    # Add the "Next Video" button
    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn = Button(ax_button, 'Next Video', color='gray', hovercolor='white')
    btn.on_clicked(pick_new_video)

    print("\nVisualizer launched! Playing skeletons at 30 FPS.")
    plt.show()

if __name__ == "__main__":
    launch_visualizer()