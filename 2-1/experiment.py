import math
import os
import struct
import zlib
from typing import List, Tuple

Pixel = Tuple[int, int, int]


def clamp(value: float, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(round(value))))


def create_image(width: int, height: int, color: Pixel = (255, 255, 255)) -> List[List[Pixel]]:
    return [[color for _ in range(width)] for _ in range(height)]


def write_png(pixels: List[List[Pixel]], path: str) -> None:
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    raw = bytearray()
    for row in pixels:
        raw.append(0)  # filter type 0
        for r, g, b in row:
            raw.extend([clamp(r), clamp(g), clamp(b)])
    compressor = zlib.compress(raw, 9)

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


def draw_rectangle(pixels: List[List[Pixel]], top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Pixel) -> None:
    x0, y0 = top_left
    x1, y1 = bottom_right
    for y in range(max(0, y0), min(len(pixels), y1)):
        for x in range(max(0, x0), min(len(pixels[0]), x1)):
            pixels[y][x] = color


def draw_circle(pixels: List[List[Pixel]], center: Tuple[int, int], radius: int, color: Pixel) -> None:
    cx, cy = center
    for y in range(len(pixels)):
        for x in range(len(pixels[0])):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                pixels[y][x] = color


def draw_triangle(pixels: List[List[Pixel]], p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int], color: Pixel) -> None:
    # Barycentric fill
    def edge(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    min_x = max(0, min(p1[0], p2[0], p3[0]))
    max_x = min(len(pixels[0]) - 1, max(p1[0], p2[0], p3[0]))
    min_y = max(0, min(p1[1], p2[1], p3[1]))
    max_y = min(len(pixels) - 1, max(p1[1], p2[1], p3[1]))
    area = edge(p1, p2, p3)
    if area == 0:
        return
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = (x, y)
            w1 = edge(p2, p3, p)
            w2 = edge(p3, p1, p)
            w3 = edge(p1, p2, p)
            if (w1 >= 0 and w2 >= 0 and w3 >= 0) or (w1 <= 0 and w2 <= 0 and w3 <= 0):
                pixels[y][x] = color


def generate_test_image(size: int = 256) -> List[List[Pixel]]:
    img = create_image(size, size, (240, 240, 240))
    draw_rectangle(img, (30, 30), (120, 200), (200, 60, 60))
    draw_circle(img, (180, 90), 50, (60, 120, 220))
    draw_triangle(img, (160, 170), (230, 210), (200, 120), (70, 180, 80))
    # Add grid lines
    for y in range(0, size, 16):
        for x in range(size):
            img[y][x] = (100, 100, 100)
    for x in range(0, size, 16):
        for y in range(size):
            img[y][x] = (100, 100, 100)
    return img


def mat_mult(T: List[List[float]], vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x = T[0][0] * vec[0] + T[0][1] * vec[1] + T[0][2] * vec[2]
    y = T[1][0] * vec[0] + T[1][1] * vec[1] + T[1][2] * vec[2]
    z = T[2][0] * vec[0] + T[2][1] * vec[1] + T[2][2] * vec[2]
    return x, y, z


def invert_matrix(T: List[List[float]]) -> List[List[float]]:
    a, b, c = T[0]
    d, e, f = T[1]
    g, h, i = T[2]
    det = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(det) < 1e-9:
        raise ValueError("Matrix is singular and cannot be inverted")
    inv_det = 1.0 / det
    return [
        [ (e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det ],
        [ (f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det ],
        [ (d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det ],
    ]


def get_pixel(pixels: List[List[Pixel]], x: int, y: int) -> Pixel:
    height = len(pixels)
    width = len(pixels[0])
    if 0 <= x < width and 0 <= y < height:
        return pixels[y][x]
    return (0, 0, 0)


def bilinear_interpolate(pixels: List[List[Pixel]], x: float, y: float) -> Pixel:
    x0 = math.floor(x)
    x1 = x0 + 1
    y0 = math.floor(y)
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0
    p00 = get_pixel(pixels, x0, y0)
    p10 = get_pixel(pixels, x1, y0)
    p01 = get_pixel(pixels, x0, y1)
    p11 = get_pixel(pixels, x1, y1)
    result = []
    for c in range(3):
        top = (1 - dx) * p00[c] + dx * p10[c]
        bottom = (1 - dx) * p01[c] + dx * p11[c]
        val = (1 - dy) * top + dy * bottom
        result.append(clamp(val))
    return tuple(result)  # type: ignore


def forward_transform(pixels: List[List[Pixel]], T: List[List[float]], out_size: Tuple[int, int]) -> List[List[Pixel]]:
    out_w, out_h = out_size
    output = create_image(out_w, out_h, (0, 0, 0))
    for y in range(len(pixels)):
        for x in range(len(pixels[0])):
            dx, dy, dz = mat_mult(T, (x, y, 1))
            if abs(dz) < 1e-9:
                continue
            dx /= dz
            dy /= dz
            ix = int(round(dx))
            iy = int(round(dy))
            if 0 <= ix < out_w and 0 <= iy < out_h:
                output[iy][ix] = pixels[y][x]
    return output


def inverse_transform(pixels: List[List[Pixel]], T_inv: List[List[float]], out_size: Tuple[int, int], interpolation: str = "nearest") -> List[List[Pixel]]:
    out_w, out_h = out_size
    output = create_image(out_w, out_h, (0, 0, 0))
    for y in range(out_h):
        for x in range(out_w):
            sx, sy, sz = mat_mult(T_inv, (x, y, 1))
            if abs(sz) < 1e-9:
                continue
            sx /= sz
            sy /= sz
            if interpolation == "nearest":
                ix = int(round(sx))
                iy = int(round(sy))
                output[y][x] = get_pixel(pixels, ix, iy)
            else:
                output[y][x] = bilinear_interpolate(pixels, sx, sy)
    return output


def gaussian_kernel() -> List[List[float]]:
    base = [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]
    return [[v / 256.0 for v in row] for row in base]


def convolve(pixels: List[List[Pixel]], kernel: List[List[float]]) -> List[List[Pixel]]:
    kh = len(kernel)
    kw = len(kernel[0])
    pad_h = kh // 2
    pad_w = kw // 2
    height = len(pixels)
    width = len(pixels[0])
    output = create_image(width, height, (0, 0, 0))
    for y in range(height):
        for x in range(width):
            accum = [0.0, 0.0, 0.0]
            for ky in range(kh):
                for kx in range(kw):
                    ny = y + ky - pad_h
                    nx = x + kx - pad_w
                    px = get_pixel(pixels, nx, ny)
                    k = kernel[ky][kx]
                    accum[0] += px[0] * k
                    accum[1] += px[1] * k
                    accum[2] += px[2] * k
            output[y][x] = (clamp(accum[0]), clamp(accum[1]), clamp(accum[2]))
    return output


def downsample(pixels: List[List[Pixel]], factor: int = 2) -> List[List[Pixel]]:
    blurred = convolve(pixels, gaussian_kernel())
    new_h = len(blurred) // factor
    new_w = len(blurred[0]) // factor
    output = create_image(new_w, new_h, (0, 0, 0))
    for y in range(new_h):
        for x in range(new_w):
            output[y][x] = blurred[y * factor][x * factor]
    return output


def upsample(pixels: List[List[Pixel]], target_size: Tuple[int, int]) -> List[List[Pixel]]:
    target_w, target_h = target_size
    enlarged = create_image(target_w, target_h, (0, 0, 0))
    for y in range(len(pixels)):
        for x in range(len(pixels[0])):
            ty = y * 2
            tx = x * 2
            if ty < target_h and tx < target_w:
                enlarged[ty][tx] = pixels[y][x]
    return convolve(enlarged, gaussian_kernel())


def subtract_images(a: List[List[Pixel]], b: List[List[Pixel]]) -> List[List[Pixel]]:
    height = len(a)
    width = len(a[0])
    output = create_image(width, height, (0, 0, 0))
    for y in range(height):
        for x in range(width):
            diff = [a[y][x][c] - b[y][x][c] for c in range(3)]
            output[y][x] = (diff[0], diff[1], diff[2])
    return output


def visualize_laplacian(pixels: List[List[Pixel]]) -> List[List[Pixel]]:
    """Normalize Laplacian values to a visible 0-255 range."""

    height = len(pixels)
    width = len(pixels[0])

    # Find the symmetric range around zero to avoid bright bias
    max_abs = [0.0, 0.0, 0.0]
    for y in range(height):
        for x in range(width):
            for c in range(3):
                max_abs[c] = max(max_abs[c], abs(pixels[y][x][c]))

    # Prevent division by zero
    max_abs = [m if m > 1e-6 else 1.0 for m in max_abs]

    output = create_image(width, height, (0, 0, 0))
    for y in range(height):
        for x in range(width):
            output[y][x] = (
                clamp((pixels[y][x][0] / (2 * max_abs[0]) + 0.5) * 255),
                clamp((pixels[y][x][1] / (2 * max_abs[1]) + 0.5) * 255),
                clamp((pixels[y][x][2] / (2 * max_abs[2]) + 0.5) * 255),
            )
    return output


def build_gaussian_pyramid(pixels: List[List[Pixel]], levels: int) -> List[List[List[Pixel]]]:
    pyramid = [pixels]
    for _ in range(1, levels):
        pyramid.append(downsample(pyramid[-1]))
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid: List[List[List[Pixel]]]) -> List[List[List[Pixel]]]:
    laplacian = []
    for level in range(len(gaussian_pyramid) - 1):
        expanded = upsample(gaussian_pyramid[level + 1], (len(gaussian_pyramid[level][0]), len(gaussian_pyramid[level])))
        laplacian.append(subtract_images(gaussian_pyramid[level], expanded))
    laplacian.append(gaussian_pyramid[-1])
    return laplacian


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image(pixels: List[List[Pixel]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    write_png(pixels, path)
    print(f"Saved {path}")


def main() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    ensure_dir(output_dir)
    base = generate_test_image()
    save_image(base, os.path.join(output_dir, "original.png"))

    center_x = len(base[0]) / 2
    center_y = len(base) / 2

    # Transform definitions
    transforms = {
        "translation": [
            [1, 0, 40],
            [0, 1, 30],
            [0, 0, 1],
        ],
        "rotation": [
            [math.cos(math.radians(30)), -math.sin(math.radians(30)), center_x - center_x * math.cos(math.radians(30)) + center_y * math.sin(math.radians(30))],
            [math.sin(math.radians(30)), math.cos(math.radians(30)), center_y - center_x * math.sin(math.radians(30)) - center_y * math.cos(math.radians(30))],
            [0, 0, 1],
        ],
        "euclidean": [
            [math.cos(math.radians(-20)), -math.sin(math.radians(-20)), 30],
            [math.sin(math.radians(-20)), math.cos(math.radians(-20)), -10],
            [0, 0, 1],
        ],
        "similarity": [
            [0.85 * math.cos(math.radians(25)), -0.85 * math.sin(math.radians(25)), 20],
            [0.85 * math.sin(math.radians(25)), 0.85 * math.cos(math.radians(25)), 15],
            [0, 0, 1],
        ],
        "affine": [
            [1.1, 0.2, -20],
            [0.1, 0.9, 15],
            [0, 0, 1],
        ],
    }

    out_size = (len(base[0]), len(base))

    for name, T in transforms.items():
        T_inv = invert_matrix(T)
        forward_img = forward_transform(base, T, out_size)
        inverse_nearest = inverse_transform(base, T_inv, out_size, interpolation="nearest")
        inverse_bilinear = inverse_transform(base, T_inv, out_size, interpolation="bilinear")

        save_image(forward_img, os.path.join(output_dir, f"forward_{name}.png"))
        save_image(inverse_nearest, os.path.join(output_dir, f"inverse_nearest_{name}.png"))
        save_image(inverse_bilinear, os.path.join(output_dir, f"inverse_bilinear_{name}.png"))

    gaussian = build_gaussian_pyramid(base, 4)
    for idx, level in enumerate(gaussian):
        save_image(level, os.path.join(output_dir, f"gaussian_level_{idx}.png"))

    laplacian = build_laplacian_pyramid(gaussian)
    for idx, level in enumerate(laplacian):
        vis = visualize_laplacian(level)
        save_image(vis, os.path.join(output_dir, f"laplacian_level_{idx}.png"))


if __name__ == "__main__":
    main()
