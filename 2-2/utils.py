import math
import os
import struct
import zlib
from typing import List, Tuple

Pixel = Tuple[int, int, int]
GrayImage = List[List[float]]
ColorImage = List[List[Pixel]]


def clamp(value: float, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(round(value))))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_png(image: ColorImage, path: str) -> None:
    height = len(image)
    width = len(image[0]) if height else 0
    raw = bytearray()
    for row in image:
        raw.append(0)  # filter type 0
        for r, g, b in row:
            raw.extend([clamp(r), clamp(g), clamp(b)])
    compressor = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", compressor) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def save_gray(image: GrayImage, path: str, normalize: bool = True) -> None:
    flat = [v for row in image for v in row]
    if normalize:
        mn = min(flat)
        mx = max(flat)
        if mx > mn:
            flat = [(v - mn) / (mx - mn) * 255.0 for v in flat]
        else:
            flat = [0.0 for _ in flat]
    else:
        flat = [max(0.0, min(255.0, v)) for v in flat]
    idx = 0
    rgb_rows: ColorImage = []
    for y in range(len(image)):
        row: List[Pixel] = []
        for x in range(len(image[0])):
            val = clamp(flat[idx])
            idx += 1
            row.append((val, val, val))
        rgb_rows.append(row)
    write_png(rgb_rows, path)


def hsv_to_rgb(h: float, s: float, v: float) -> Pixel:
    h = (h % 360) / 60.0
    c = v * s
    x = c * (1 - abs(h % 2 - 1))
    m = v - c
    if 0 <= h < 1:
        r, g, b = c, x, 0
    elif 1 <= h < 2:
        r, g, b = x, c, 0
    elif 2 <= h < 3:
        r, g, b = 0, c, x
    elif 3 <= h < 4:
        r, g, b = 0, x, c
    elif 4 <= h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (clamp((r + m) * 255), clamp((g + m) * 255), clamp((b + m) * 255))


def to_orientation_rgb(theta: GrayImage) -> ColorImage:
    height = len(theta)
    width = len(theta[0]) if height else 0
    img: ColorImage = []
    for y in range(height):
        row: List[Pixel] = []
        for x in range(width):
            h = (theta[y][x] + math.pi) / (2 * math.pi) * 360.0
            row.append(hsv_to_rgb(h, 1.0, 1.0))
        img.append(row)
    return img


def gaussian_kernel(sigma: float, radius: int | None = None) -> List[List[float]]:
    if radius is None:
        radius = int(math.ceil(3 * sigma))
    size = radius * 2 + 1
    kernel: List[List[float]] = []
    total = 0.0
    for y in range(size):
        row: List[float] = []
        for x in range(size):
            yy = y - radius
            xx = x - radius
            val = math.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)) / (2 * math.pi * sigma * sigma)
            row.append(val)
            total += val
        kernel.append(row)
    for y in range(size):
        for x in range(size):
            kernel[y][x] /= total
    return kernel


def gaussian_derivative_kernel(sigma: float, axis: str, radius: int | None = None) -> List[List[float]]:
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if radius is None:
        radius = int(math.ceil(3 * sigma))
    size = radius * 2 + 1
    kernel: List[List[float]] = []
    mean = 0.0
    for y in range(size):
        row: List[float] = []
        for x in range(size):
            yy = y - radius
            xx = x - radius
            factor = -xx if axis == "x" else -yy
            val = factor * math.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma)) / (2 * math.pi * sigma ** 4)
            row.append(val)
            mean += val
        kernel.append(row)
    mean /= size * size
    for y in range(size):
        for x in range(size):
            kernel[y][x] -= mean
    return kernel


def convolve(image: GrayImage, kernel: List[List[float]]) -> GrayImage:
    kh = len(kernel)
    kw = len(kernel[0])
    pad_h = kh // 2
    pad_w = kw // 2
    height = len(image)
    width = len(image[0]) if height else 0
    out: GrayImage = [[0.0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            acc = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    iy = min(max(y + ky - pad_h, 0), height - 1)
                    ix = min(max(x + kx - pad_w, 0), width - 1)
                    acc += image[iy][ix] * kernel[ky][kx]
            out[y][x] = acc
    return out


def gradient_from_derivative(image: GrayImage, sigma: float) -> Tuple[GrayImage, GrayImage]:
    kx = gaussian_derivative_kernel(sigma, "x")
    ky = gaussian_derivative_kernel(sigma, "y")
    gx = convolve(image, kx)
    gy = convolve(image, ky)
    return gx, gy


def gradient_magnitude_orientation(gx: GrayImage, gy: GrayImage) -> Tuple[GrayImage, GrayImage]:
    height = len(gx)
    width = len(gx[0]) if height else 0
    mag: GrayImage = []
    theta: GrayImage = []
    for y in range(height):
        mag_row: List[float] = []
        theta_row: List[float] = []
        for x in range(width):
            mx = math.hypot(gx[y][x], gy[y][x])
            mag_row.append(mx)
            theta_row.append(math.atan2(gy[y][x], gx[y][x]))
        mag.append(mag_row)
        theta.append(theta_row)
    return mag, theta


def create_test_image(size: int = 256) -> ColorImage:
    img: ColorImage = [[(240, 240, 240) for _ in range(size)] for _ in range(size)]
    for y in range(30, 200):
        for x in range(30, 120):
            img[y][x] = (200, 60, 60)
    for y in range(size):
        for x in range(size):
            if (x - 180) ** 2 + (y - 90) ** 2 <= 50 ** 2:
                img[y][x] = (60, 120, 220)
    for y in range(120, 211):
        for x in range(size):
            if y - 120 <= x - 160 and y - 120 <= -(x - 200) * 90 / 40 + 90 and x >= 160 and x <= 230:
                img[y][x] = (70, 180, 80)
    for y in range(0, size, 16):
        for x in range(size):
            img[y][x] = (100, 100, 100)
    for x in range(0, size, 16):
        for y in range(size):
            img[y][x] = (100, 100, 100)
    return img


def rgb_to_gray(image: ColorImage) -> GrayImage:
    height = len(image)
    width = len(image[0]) if height else 0
    gray: GrayImage = []
    for y in range(height):
        row: List[float] = []
        for x in range(width):
            r, g, b = image[y][x]
            row.append(0.299 * r + 0.587 * g + 0.114 * b)
        gray.append(row)
    return gray


def sobel_kernels() -> Tuple[List[List[float]], List[List[float]]]:
    kx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    return kx, ky


def resize_nearest(image: ColorImage, scale: float) -> ColorImage:
    src_h = len(image)
    src_w = len(image[0]) if src_h else 0
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    out: ColorImage = []
    for y in range(new_h):
        row: List[Pixel] = []
        src_y = min(int(round(y / scale)), src_h - 1)
        for x in range(new_w):
            src_x = min(int(round(x / scale)), src_w - 1)
            row.append(image[src_y][src_x])
        out.append(row)
    return out


def rotate_image(image: ColorImage, angle_deg: float) -> ColorImage:
    angle = math.radians(angle_deg)
    src_h = len(image)
    src_w = len(image[0]) if src_h else 0
    cy = (src_h - 1) / 2.0
    cx = (src_w - 1) / 2.0
    out: ColorImage = []
    for y in range(src_h):
        row: List[Pixel] = []
        for x in range(src_w):
            x0 = (x - cx) * math.cos(-angle) - (y - cy) * math.sin(-angle) + cx
            y0 = (x - cx) * math.sin(-angle) + (y - cy) * math.cos(-angle) + cy
            ix = min(max(int(round(x0)), 0), src_w - 1)
            iy = min(max(int(round(y0)), 0), src_h - 1)
            row.append(image[iy][ix])
        out.append(row)
    return out

