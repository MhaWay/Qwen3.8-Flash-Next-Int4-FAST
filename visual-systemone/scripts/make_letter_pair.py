#!/usr/bin/env python3
"""Generate a controlled B/D JPEG pair for held-out visual-state probes."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path(__file__).resolve().parents[2]
    font = ImageFont.load_default(size=340)
    for letter in ("B", "D"):
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        draw.text((256, 256), letter, font=font, anchor="mm", fill="black",
                  stroke_width=5, stroke_fill="black")
        path = root / f"systemone-{letter}-clear.jpg"
        image.save(path, "JPEG", quality=95, subsampling=0, optimize=True)
        print(path.name)


if __name__ == "__main__":
    main()
