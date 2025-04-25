import os
import subprocess
from PIL import Image
import numpy as np
from tqdm import tqdm

def get_video_duration(video_path):
    """Returns video duration in seconds."""
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return float(result.stdout.strip())

def extract_frame_at(video_path, timestamp, output_path):
    """Extracts a single frame at the given timestamp using ffmpeg."""
    subprocess.run(
        [
            'ffmpeg', '-y',
            '-ss', f'{timestamp:.3f}',
            '-i', video_path,
            '-frames:v', '1',
            output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def get_dominant_color(image_path, resize_to=(100, 100)):
    """Find the dominant color of an image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(resize_to, resample=Image.NEAREST)
    arr = np.array(img).reshape(-1, 3)
    colors, counts = np.unique(arr, axis=0, return_counts=True)
    return tuple(colors[np.argmax(counts)])

def movie_palette(
    video_path, 
    output_path,
    short_side=200,     # user: set shorter image side, e.g. height if ratio > 1
    ratio=(5,2),        # (width, height), e.g., (5,2)
    n_bars=400          # number of vertical bars
):
    # Compute image size
    if ratio[0] > ratio[1]:
        height = short_side
        width = int(height * (ratio[0] / ratio[1]))
    else:
        width = short_side
        height = int(width * (ratio[1] / ratio[0]))
    width = n_bars   # Override width with user number of bars
    print(f"Output image size: {width}x{height}")

    # 1. Get video duration
    duration = get_video_duration(video_path)
    print(f"Video duration: {duration:.2f} seconds")

    # 2. Calculate sample points
    times = np.linspace(0, duration, width, endpoint=False)

    # 3. Process frames
    tmp_dir = 'movie_palette_tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    palette = np.zeros((height, width, 3), dtype=np.uint8)

    print("Extracting frames and computing dominant colors...")
    for i, t in enumerate(tqdm(times, desc="Processing")):
        frame_path = os.path.join(tmp_dir, f'frame_{i:04d}.jpg')
        extract_frame_at(video_path, t, frame_path)
        dom_color = get_dominant_color(frame_path)
        palette[:, i] = dom_color
        os.remove(frame_path)

    # 4. Save the result
    result_img = Image.fromarray(palette, 'RGB')
    result_img.save(output_path)
    print(f"Saved: {output_path}")

    # 5. Cleanup
    os.rmdir(tmp_dir)

if __name__ == "__main__":
    # Example usage:
    # Adjust these as needed!
    video_file = 'alfabeets.mp4'
    output_image = 'movie_palette.png'
    short_side = 200
    aspect_ratio = (5, 2)
    n_bars = 400   # or 1000 for finer

    movie_palette(video_file, output_image, short_side, aspect_ratio, n_bars)
