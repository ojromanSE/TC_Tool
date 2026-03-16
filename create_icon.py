"""
Convert icon.png to icon.ico for SE_Tool.
Run once before building: python create_icon.py
"""
from PIL import Image
import os

src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")

img = Image.open(src).convert("RGBA")
img.save(
    out,
    format="ICO",
    sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
)
print(f"Saved {out}")
