import os
import shutil
import argparse
import subprocess
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

from sklearn.cluster import KMeans


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


def filter_pixels(arr, dark_threshold=30, white_threshold=225):
    """Return pixels not too dark and not too bright."""
    not_black = np.any(arr > dark_threshold, axis=1)
    not_white = np.any(arr < white_threshold, axis=1)
    mask = not_black & not_white
    filtered_arr = arr[mask]
    if filtered_arr.size == 0:
        filtered_arr = arr
    return filtered_arr

def dominant_color(filtered_arr):
    colors, counts = np.unique(filtered_arr, axis=0, return_counts=True)
    return tuple(colors[np.argmax(counts)])

def average_color(filtered_arr):
    return tuple(np.mean(filtered_arr, axis=0).astype(int))

def median_color(filtered_arr):
    return tuple(np.median(filtered_arr, axis=0).astype(int))

def kmeans_color(filtered_arr, k=3, mode="biggest"):
    if KMeans is None:
        raise ImportError("scikit-learn not installed")

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(filtered_arr)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_

    if mode == "biggest":
        _, counts = np.unique(labels, return_counts=True)
        idx = np.argmax(counts)
    elif mode == "brightest":
        # Convert to HSV and get the one with max value (brightness)
        hsv = np.array([cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0][0] for c in centers])
        idx = np.argmax(hsv[:, 2])  # Value (brightness)
    elif mode == "saturated":
        hsv = np.array([cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0][0] for c in centers])
        idx = np.argmax(hsv[:, 1])  # Saturation
    else:
        raise ValueError("Invalid mode. Use 'biggest', 'brightest', or 'saturated'.")

    return tuple(map(int, centers[idx]))

def most_saturated_color(filtered_arr, min_value=64):
    """
    Pick the pixel with highest saturation, but only among those with V >= min_value (in HSV).
    """
    if cv2 is None:
        raise ImportError("cv2 (opencv-python) not installed")
    arr = filtered_arr.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).reshape(-1,3)
    # Only consider pixels above brightness threshold
    candidates = hsv[:,2] >= min_value
    if np.any(candidates):
        idx = np.argmax(hsv[candidates,1])
        idx_all = np.where(candidates)[0][idx]
    else:
        # fallback to any pixel if none are bright enough
        idx_all = np.argmax(hsv[:,1])
    return tuple(filtered_arr[idx_all])

def get_color_by_algorithm(image_path, resize_to=(100,100), dark_threshold=30, white_threshold=225, algo="dominant", kmeans_k=3, kmeans_mode="biggest"):
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize(resize_to, resample=Image.NEAREST)
    arr = np.array(img).reshape(-1, 3)
    filtered_arr = filter_pixels(arr, dark_threshold, white_threshold)
    if algo == "dominant":
        return dominant_color(filtered_arr)
    elif algo == "average":
        return average_color(filtered_arr)
    elif algo == "median":
        return median_color(filtered_arr)
    elif algo == "kmeans":
        return kmeans_color(filtered_arr, k=kmeans_k, mode=kmeans_mode)
    elif algo == "saturated":
        return most_saturated_color(filtered_arr)
    else:
        raise ValueError(f"Unknown color algorithm: {algo}")


def get_dominant_color(image_path, resize_to=(100, 100), dark_threshold=30, white_threshold=225):
    """Find the dominant color, ignoring very dark pixels."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(resize_to, resample=Image.NEAREST)
    arr = np.array(img).reshape(-1, 3)
    # Filter out pixels where all channels are below threshold
    not_black = np.any(arr > dark_threshold, axis=1)
    not_white = np.any(arr < white_threshold, axis=1)
    mask = not_black & not_white
    filtered_arr = arr[mask]
    # Fallback if everything is filtered out
    if filtered_arr.size == 0:
        filtered_arr = arr
    colors, counts = np.unique(filtered_arr, axis=0, return_counts=True)
    return tuple(colors[np.argmax(counts)])


def compute_palette_size(ratio, short_side, n_bars):
    """
    Compute (height, width, bar_width) so that:
    - Image aspect ratio is fixed (ratio[0]:ratio[1])
    - Shorter side is at least short_side
    - bar_width >= 1 px (bars will be as thick as possible)
    """
    height_ratio, width_ratio = ratio
    # Assume landscape: short_side is height, long side is width
    height = short_side
    width = int(round(height * (width_ratio / height_ratio)))
    bar_width = width // n_bars

    if bar_width < 1:
        # Too many bars: set bar_width = 1, width = n_bars
        bar_width = 1
        width = n_bars * bar_width
        height = int(round(width * (height_ratio / width_ratio)))
        if height < short_side:
            height = short_side
            width = int(round(height * (width_ratio / height_ratio)))
    return int(height), int(width), int(bar_width)


def movie_palette(
    video_path, 
    output_path,
    short_side=200,
    ratio=(2,5),
    n_bars=400,
    dark_threshold=30,
    white_threshold=225,
    kmeans_k=3,
    kmeans_mode="biggest",
    color_algo="dominant"
   
):
    height, width, bar_width = compute_palette_size(ratio, short_side, n_bars)
    print(f"Output image size: {width}x{height}, Bar width: {bar_width}px, Bars: {n_bars}")

    duration = get_video_duration(video_path)
    print(f"Video duration: {duration:.2f} seconds")
    times = np.linspace(0, duration, n_bars, endpoint=False)
    tmp_dir = 'movie_palette_tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    palette = np.zeros((height, width, 3), dtype=np.uint8)

    print("Extracting frames and computing colors...")
    for i, t in enumerate(tqdm(times, desc="Processing")):
        frame_path = os.path.join(tmp_dir, f'frame_{i:04d}.jpg')
        extract_frame_at(video_path, t, frame_path)
        color = get_color_by_algorithm(
            frame_path, 
            dark_threshold=dark_threshold,
            white_threshold=white_threshold,
            algo=color_algo,
            kmeans_k=kmeans_k, 
            kmeans_mode=kmeans_mode
        )
        palette[:, i*bar_width:(i+1)*bar_width] = color
        os.remove(frame_path)

    result_img = Image.fromarray(palette, 'RGB')
    result_img.save(output_path)
    print(f"Saved: {output_path}")
    shutil.rmtree(tmp_dir)

def parse_ratio(ratio_str):
    """Parse ratio string like '5:2' to tuple (5, 2)"""
    parts = ratio_str.split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Ratio must be in format W:H (e.g., 2:5)")
    return (float(parts[0]), float(parts[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a movie color palette image from a video file.")
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("output_path", help="Path to save the palette image.")
    parser.add_argument("--short-side", type=int, default=200, help="Shorter side of output image (default: 200).")
    parser.add_argument("--ratio", type=parse_ratio, default=(2,5), help="Aspect ratio H:W (default: 2:5).")
    parser.add_argument("--n-bars", type=int, default=400, help="Number of color bars (default: 400).")
    parser.add_argument("--dark-threshold", type=int, default=30, help="Ignore pixels darker than this value (default: 30)")
    parser.add_argument("--white-threshold", type=int, default=225, help="Ignore pixels brighter than this value (default: 225)")
    parser.add_argument("--color-algo", choices=["dominant", "average", "median", "kmeans", "saturated"], default="dominant", help="Algorithm for picking color for each frame: dominant, average, median, kmeans, saturated (default: dominant)")
    parser.add_argument("--kmeans-mode", choices=["biggest", "brightest", "saturated"], default="biggest", help="Which k-means cluster to choose: biggest (default), brightest, or saturated")
    parser.add_argument("--kmeans-k", type=int, default=3, help="Number of clusters to use in k-means color extraction (default: 3)")


    args = parser.parse_args()

    movie_palette(
        args.video_path,
        args.output_path,
        args.short_side,
        args.ratio,
        args.n_bars,
        args.dark_threshold,
        args.white_threshold,
        args.kmeans_k,
        args.kmeans_mode,
        args.color_algo
    )