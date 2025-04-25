# Movie Palette Generator

Generate a **movie palette** (film barcode) where each vertical stripe shows the dominant color of a sampled frame from a movie.  
Inspired by classic “film barcode” art, this tool captures a film’s color mood in a single image.

---

## Features

- **Automatic video probing:** Detects video length with `ffprobe`.
- **Custom aspect ratio:** Specify aspect ratio as Height:Width (e.g., `2:5` for a wide palette).
- **Minimum shorter side:** Always respects your chosen minimum for the short side (height for wide palettes).
- **Dynamic bar width:** Fewer frames = wider bars; many frames = thinner bars (minimum 1px). If you choose a large number of bars, the image will grow to fit.
- **Frame sampling:** Evenly samples frames across the entire movie duration.
- **Dominant color extraction:** Finds the most frequent color in each sampled frame, **ignoring very dark and very bright pixels** (customizable).
- **Fast and easy:** Uses Python, Pillow, NumPy, tqdm, and ffmpeg.

---

## Example Output

![movie_palette](https://github.com/user-attachments/assets/efa763dd-8b84-411d-8608-ebaa801b6427)

*(Color palette from this short animation movie: https://www.youtube.com/watch?v=W5e9GJfHC4A)*

---

## Requirements

- **Python 3.7+**
- **ffmpeg** and **ffprobe** installed and in your system PATH ([Download](https://ffmpeg.org/download.html))
- **Python packages:** `Pillow`, `numpy`, `tqdm`

Install dependencies:
```bash
pip install pillow numpy tqdm
```

---

## Usage

```bash
python movie_palette.py <video_path> <output_path> [options]
```

### **Options**

- `--short-side`: Minimum size for the shorter image side (default: 200).
- `--ratio`: Aspect ratio as Height:Width (default: 2:5 for wide palettes).
- `--n-bars`: Number of color bars / frames to sample (default: 400).
- `--dark-threshold`: Ignore pixels darker than this value (default: 30).
- `--white-threshold`: Ignore pixels brighter than this value (default: 225).

### **Example**

```bash
python movie_palette.py mymovie.mp4 palette.png --short-side 200 --ratio 2:5 --n-bars 400
```
- This will produce a wide palette, at least 200px high, with 400 color bars.

---

## **How Color Extraction Works**

For each sampled frame, the tool:
- **Ignores very dark pixels** (darker than `--dark-threshold`).
- **Ignores very bright pixels** (brighter than `--white-threshold`).
- Picks the most frequent remaining color as that bar's color.

This ensures the palette focuses on the film’s true colors, not black bars or white fades/credits.

---

## **Tips**

- For a classic "film barcode" look, use `--ratio 2:5`.
- If you want a tall/portrait palette, swap to `--ratio 5:2`.
- Increasing `--n-bars` creates a more detailed palette, but will grow the image size if needed to keep bars visible.

---

## License

MIT License.  
Feel free to modify and use for your own projects!

---

**Let me know if you’d like a sample output gallery or more advanced usage tips!**
