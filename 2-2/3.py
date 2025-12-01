"""
Experiment 2-2 Task 3: Harris corner detection with parameter study and invariance demos.
"""
from pathlib import Path
from typing import List, Tuple

from utils import (
    Pixel,
    convolve,
    create_test_image,
    ensure_dir,
    gaussian_kernel,
    gradient_from_derivative,
    resize_nearest,
    rotate_image,
    rgb_to_gray,
    save_gray,
    write_png,
)


def harris_response(gray: List[List[float]], sigma_grad: float, sigma_window: float, alpha: float) -> List[List[float]]:
    gx, gy = gradient_from_derivative(gray, sigma_grad)
    ix2: List[List[float]] = []
    iy2: List[List[float]] = []
    ixy: List[List[float]] = []
    for y in range(len(gx)):
        row_ix2: List[float] = []
        row_iy2: List[float] = []
        row_ixy: List[float] = []
        for x in range(len(gx[0])):
            row_ix2.append(gx[y][x] * gx[y][x])
            row_iy2.append(gy[y][x] * gy[y][x])
            row_ixy.append(gx[y][x] * gy[y][x])
        ix2.append(row_ix2)
        iy2.append(row_iy2)
        ixy.append(row_ixy)

    window = gaussian_kernel(sigma_window)
    s_ix2 = convolve(ix2, window)
    s_iy2 = convolve(iy2, window)
    s_ixy = convolve(ixy, window)

    h = len(gray)
    w = len(gray[0]) if h else 0
    response: List[List[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            det = s_ix2[y][x] * s_iy2[y][x] - s_ixy[y][x] * s_ixy[y][x]
            trace = s_ix2[y][x] + s_iy2[y][x]
            response[y][x] = det - alpha * trace * trace
    return response


def nms_corners(response: List[List[float]], threshold: float, radius: int = 1) -> List[Tuple[int, int]]:
    h = len(response)
    w = len(response[0]) if h else 0
    corners: List[Tuple[int, int, float]] = []
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            if response[y][x] < threshold:
                continue
            max_patch = max(
                response[yy][xx]
                for yy in range(y - radius, y + radius + 1)
                for xx in range(x - radius, x + radius + 1)
            )
            if response[y][x] == max_patch:
                corners.append((y, x, response[y][x]))
    corners.sort(key=lambda c: c[2], reverse=True)
    return [(y, x) for y, x, _ in corners]


def draw_corners(image: List[List[Pixel]], corners: List[Tuple[int, int]]) -> List[List[Pixel]]:
    vis = [row.copy() for row in image]
    for y, x in corners:
        for yy in range(max(0, y - 1), min(len(vis), y + 2)):
            for xx in range(max(0, x - 1), min(len(vis[0]), x + 2)):
                vis[yy][xx] = (220, 40, 40)
    return vis


def max_value(img: List[List[float]]) -> float:
    return max(v for row in img for v in row)


def run(
    sigma_grad: float = 1.0,
    window_sigmas: Tuple[float, ...] = (0.8, 1.5, 2.5),
    alpha: float = 0.05,
    threshold_ratio: float = 0.01,
    scale_factor: float = 0.7,
    rotation_deg: float = 30.0,
) -> None:
    base_dir = Path(__file__).parent
    out_dir = base_dir / "outputs" / "task3"
    ensure_dir(str(out_dir))

    rgb = create_test_image()
    gray = rgb_to_gray(rgb)
    write_png(rgb, out_dir / "input.png")

    last_resp = None
    last_corners = None
    last_sigma_window = window_sigmas[-1]
    for sigma_window in window_sigmas:
        resp = harris_response(gray, sigma_grad, sigma_window, alpha)
        thresh = max_value(resp) * threshold_ratio
        corners = nms_corners(resp, thresh, radius=2)
        save_gray(resp, out_dir / f"response_window{sigma_window:.2f}.png", normalize=True)
        write_png(draw_corners(rgb, corners), out_dir / f"corners_window{sigma_window:.2f}.png")
        print(
            f"Harris window sweep: sigma_grad={sigma_grad}, sigma_window={sigma_window}, corners={len(corners)}, threshold={thresh:.2f}"
        )
        last_resp = resp
        last_corners = corners
        last_sigma_window = sigma_window

    # Scale invariance check
    scaled = resize_nearest(rgb, scale_factor)
    scaled_gray = rgb_to_gray(scaled)
    resp_scaled = harris_response(scaled_gray, sigma_grad, last_sigma_window, alpha)
    corners_scaled = nms_corners(resp_scaled, max_value(resp_scaled) * threshold_ratio, radius=2)
    write_png(draw_corners(scaled, corners_scaled), out_dir / "corners_scaled.png")

    # Rotation equivariance check
    rotated = rotate_image(rgb, rotation_deg)
    rotated_gray = rgb_to_gray(rotated)
    resp_rot = harris_response(rotated_gray, sigma_grad, last_sigma_window, alpha)
    corners_rot = nms_corners(resp_rot, max_value(resp_rot) * threshold_ratio, radius=2)
    write_png(draw_corners(rotated, corners_rot), out_dir / "corners_rotated.png")

    print(
        f"Harris invariance check: base={len(last_corners)}, scaled={len(corners_scaled)}, "
        f"rotated={len(corners_rot)}, sigma_window={last_sigma_window}"
    )


if __name__ == "__main__":
    run()
