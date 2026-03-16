"""
Generate icon.ico for SE_Tool.
Run once before building: python create_icon.py
Replace icon.ico with your own 256x256 .ico file if desired.
"""
from PIL import Image, ImageDraw, ImageFont
import struct, io, os

SIZE = 256

def make_image(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    pad = int(size * 0.06)
    r = int(size * 0.12)

    # Background rounded rect
    bg = (30, 60, 90)  # dark navy
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=r, fill=bg)

    # Accent bar at top
    accent = (0, 180, 150)  # teal
    bar_h = int(size * 0.09)
    d.rounded_rectangle([pad, pad, size - pad, pad + bar_h + r],
                        radius=r, fill=accent)
    d.rectangle([pad, pad + r, size - pad, pad + bar_h], fill=accent)

    # "SE" text
    cx, cy = size // 2, int(size * 0.56)
    font_size = int(size * 0.40)
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    bbox = d.textbbox((0, 0), "SE", font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    d.text((cx - tw // 2, cy - th // 2), "SE", fill="white", font=font)

    return img


def save_ico(path):
    sizes = [256, 128, 64, 48, 32, 16]
    images = [make_image(s) for s in sizes]
    images[0].save(
        path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=images[1:],
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "icon.ico")
    save_ico(out)
