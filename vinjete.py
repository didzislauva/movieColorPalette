import os
import argparse
from PIL import Image
import numpy as np

def apply_vertical_vignette(palette_img, strength=0.6, curve=2.0):
    """
    Apply a vertical vignette effect to an RGB image.
    
    Args:
        palette_img: PIL.Image or np.ndarray of shape (H, W, 3)
        strength: How strong the darkening is at the edges (0 = black, 1 = no effect; typical 0.5–0.8)
        curve: How fast vignette fades; 1=linear, >1=softer center
        
    Returns:
        A new PIL.Image with vignette applied.
    """
    if isinstance(palette_img, Image.Image):
        arr = np.array(palette_img).astype(np.float32)
    else:
        arr = palette_img.astype(np.float32)
    
    h, w = arr.shape[:2]
    y = np.linspace(-1, 1, h)
    vignette_profile = 1 - (1-strength) * (np.abs(y) ** curve)
    vignette_profile = vignette_profile[:, None]  # Shape (H,1)
    vignette = np.tile(vignette_profile, (1, w))
    
    arr_vig = arr * vignette[..., None]
    arr_vig = np.clip(arr_vig, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_vig)

def main():
    parser = argparse.ArgumentParser(description="Apply vertical vignette to an image")
    parser.add_argument("input_image", help="Input image file (palette image)")
    parser.add_argument("output_image", help="Output image file (with vignette)")
    parser.add_argument("--strength", type=float, default=0.6, help="Edge darkness strength (0 = black, 1 = no effect, default 0.6)")
    parser.add_argument("--curve", type=float, default=2.0, help="Vignette curve softness (1=linear, >1=softer, default 2.0)")
    args = parser.parse_args()

    img = Image.open(args.input_image).convert("RGB")
    img_vig = apply_vertical_vignette(img, strength=args.strength, curve=args.curve)
    img_vig.save(args.output_image)
    print(f"Saved: {args.output_image}")

if __name__ == "__main__":
    main()
