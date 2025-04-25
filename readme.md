# Movie Palette Generator

Generate a **movie palette** (film barcode), where each vertical stripe represents the dominant color of a sampled frame from a movie.  
Inspired by classic “film barcode” art, this tool summarizes a movie’s color mood in a single image.

---

## Features

- **Automatic video probing:** Detects video length with ffprobe.
- **Custom aspect ratio:** Specify width-to-height ratio (e.g., 5:2, 16:9, etc).
- **Flexible resolution:** Set shorter side (height or width), and choose the number of color stripes (columns).
- **Frame sampling:** Evenly samples frames across the entire movie duration.
- **Dominant color extraction:** Finds the most frequent color in each sampled frame.
- **Fast and easy:** Uses Python, Pillow, NumPy, tqdm, and ffmpeg.

---

## Example Output

![movie_palette](https://github.com/user-attachments/assets/efa763dd-8b84-411d-8608-ebaa801b6427)

( Color palette from this short animation movie https://www.youtube.com/watch?v=W5e9GJfHC4A)

---

## Requirements

- **Python 3.7+**
- **ffmpeg** and **ffprobe** installed and in your system PATH ([Download](https://ffmpeg.org/download.html))
- **Python packages:** `Pillow`, `numpy`, `tqdm`

Install dependencies:
```bash
pip install pillow numpy tqdm
