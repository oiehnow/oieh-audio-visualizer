"""Generate assets/icon.ico — the app's radial-bars mark on a dark rounded square.

Run:  python assets/make_icon.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

S = 512  # master size; ICO sizes are downscaled from this
ACCENT = (0x3C, 0xE6, 0xA8)   # app green
ACCENT2 = (0x1E, 0x5C, 0xFF)  # app blue
BG = (0x10, 0x10, 0x14)


def lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def main() -> None:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.rounded_rectangle([8, 8, S - 8, S - 8], radius=S * 0.22, fill=BG + (255,))

    # radial bars — the app's signature style
    cx = cy = S / 2
    n = 24
    r0 = S * 0.17
    lengths = [0.55, 0.8, 1.0, 0.7, 0.9, 0.6, 1.0, 0.75, 0.55, 0.85, 0.95, 0.65,
               0.8, 1.0, 0.6, 0.9, 0.7, 1.0, 0.55, 0.8, 0.9, 0.65, 1.0, 0.75]
    max_len = S * 0.155
    bar_w = S * 0.030
    for i in range(n):
        ang = (i / n) * 2 * math.pi - math.pi / 2
        t = i / (n - 1)
        col = lerp(ACCENT, ACCENT2, t)
        r1 = r0 + lengths[i] * max_len
        x0, y0 = cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)
        x1, y1 = cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)
        d.line([x0, y0, x1, y1], fill=col + (255,), width=round(bar_w))
        for x, y in ((x0, y0), (x1, y1)):  # round caps
            d.ellipse([x - bar_w / 2, y - bar_w / 2, x + bar_w / 2, y + bar_w / 2],
                      fill=col + (255,))

    # center note dot
    d.ellipse([cx - S * 0.045, cy - S * 0.045, cx + S * 0.045, cy + S * 0.045],
              fill=ACCENT + (255,))

    out = Path(__file__).parent
    img.save(out / "icon.png")
    img.save(out / "icon.ico",
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"wrote {out / 'icon.ico'} and icon.png")


if __name__ == "__main__":
    main()
