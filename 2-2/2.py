"""
Experiment 2-2 Task 2: Canny edge detector implementation with step outputs (pure Python).
"""
import math
from pathlib import Path
from typing import List

from utils import (
    convolve,
    create_test_image,
    ensure_dir,
    gaussian_kernel,
    rgb_to_gray,
    save_gray,
    sobel_kernels,
    to_orientation_rgb,
    write_png,
)


def non_maximum_suppression(mag: List[List[float]], theta: List[List[float]]) -> List[List[float]]:
    h = len(mag)
    w = len(mag[0]) if h else 0
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            angle = (math.degrees(theta[y][x]) + 180) % 180
            q = r = 0.0
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = mag[y][x + 1], mag[y][x - 1]
            elif 22.5 <= angle < 67.5:
                q, r = mag[y + 1][x - 1], mag[y - 1][x + 1]
            elif 67.5 <= angle < 112.5:
                q, r = mag[y + 1][x], mag[y - 1][x]
            else:
                q, r = mag[y - 1][x - 1], mag[y + 1][x + 1]
            if mag[y][x] >= q and mag[y][x] >= r:
                out[y][x] = mag[y][x]
    return out


def hysteresis(img: List[List[float]], low: float, high: float) -> List[List[float]]:
    h = len(img)
    w = len(img[0]) if h else 0
    strong = 255.0
    weak = 75.0
    res = [[0.0 for _ in range(w)] for _ in range(h)]
    queue = []
    for y in range(h):
        for x in range(w):
            if img[y][x] >= high:
                res[y][x] = strong
                queue.append((y, x))
            elif img[y][x] >= low:
                res[y][x] = weak
    while queue:
        y, x = queue.pop()
        for yy in range(max(0, y - 1), min(h, y + 2)):
            for xx in range(max(0, x - 1), min(w, x + 2)):
                if res[yy][xx] == weak:
                    res[yy][xx] = strong
                    queue.append((yy, xx))
    for y in range(h):
        for x in range(w):
            if res[y][x] != strong:
                res[y][x] = 0.0
    return res


def hypot_map(gx: List[List[float]], gy: List[List[float]]) -> List[List[float]]:
    h = len(gx)
    w = len(gx[0]) if h else 0
    mag = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            mag[y][x] = math.hypot(gx[y][x], gy[y][x])
    return mag


def atan2_map(gy: List[List[float]], gx: List[List[float]]) -> List[List[float]]:
    h = len(gx)
    w = len(gx[0]) if h else 0
    theta = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            theta[y][x] = math.atan2(gy[y][x], gx[y][x])
    return theta


def max_value(img: List[List[float]]) -> float:
    return max(v for row in img for v in row)


def positive_count(img: List[List[float]]) -> int:
    return sum(1 for row in img for v in row if v > 0)


def run(sigma: float = 1.0, low_ratio: float = 0.1, high_ratio: float = 0.25) -> None:
    base_dir = Path(__file__).parent
    out_dir = base_dir / "outputs" / "task2"
    ensure_dir(str(out_dir))

    rgb = create_test_image()
    gray = rgb_to_gray(rgb)
    write_png(rgb, str(out_dir / "input.png"))

    g_kernel = gaussian_kernel(sigma)
    blurred = convolve(gray, g_kernel)
    save_gray(blurred, str(out_dir / "blurred.png"))

    kx, ky = sobel_kernels()
    gx = convolve(blurred, kx)
    gy = convolve(blurred, ky)
    mag = hypot_map(gx, gy)
    theta = atan2_map(gy, gx)
    save_gray(mag, str(out_dir / "gradient.png"), normalize=True)
    write_png(to_orientation_rgb(theta), str(out_dir / "orientation.png"))

    nms = non_maximum_suppression(mag, theta)
    save_gray(nms, str(out_dir / "nms.png"), normalize=True)

    high = max_value(mag) * high_ratio
    low = high * low_ratio
    edges = hysteresis(nms, low, high)
    save_gray(edges, str(out_dir / "edges.png"), normalize=False)

    print(f"Canny: sigma={sigma}, high={high:.2f}, low={low:.2f}, edges={positive_count(edges)} pixels")


if __name__ == "__main__":
    run()
