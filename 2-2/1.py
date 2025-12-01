"""
Experiment 2-2 Task 1: Gaussian derivative gradients.
Generates gradient magnitude/orientation maps for multiple sigmas.
Outputs are written under outputs/ and ignored by git.
"""
from pathlib import Path

from utils import (
    create_test_image,
    ensure_dir,
    gradient_from_derivative,
    gradient_magnitude_orientation,
    rgb_to_gray,
    save_gray,
    to_orientation_rgb,
    write_png,
)


def run(sigmas: list[float]) -> None:
    base_dir = Path(__file__).parent
    out_dir = base_dir / "outputs" / "task1"
    ensure_dir(out_dir)

    rgb = create_test_image()
    gray = rgb_to_gray(rgb)
    write_png(rgb, out_dir / "input.png")

    for sigma in sigmas:
        gx, gy = gradient_from_derivative(gray, sigma)
        mag, theta = gradient_magnitude_orientation(gx, gy)

        save_gray(gx, out_dir / f"gx_sigma{sigma:.2f}.png", normalize=True)
        save_gray(gy, out_dir / f"gy_sigma{sigma:.2f}.png", normalize=True)
        save_gray(mag, out_dir / f"mag_sigma{sigma:.2f}.png", normalize=True)
        orient_rgb = to_orientation_rgb(theta)
        write_png(orient_rgb, out_dir / f"orient_sigma{sigma:.2f}.png")

        gx_vals = [v for row in gx for v in row]
        mag_vals = [v for row in mag for v in row]
        print(
            f"sigma={sigma:.2f}: gx [{min(gx_vals):.2f},{max(gx_vals):.2f}] mag mean {sum(mag_vals)/len(mag_vals):.2f}"
        )


if __name__ == "__main__":
    run([0.5, 1.0, 2.0])
