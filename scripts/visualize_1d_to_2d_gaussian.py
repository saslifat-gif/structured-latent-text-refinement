"""Small visual for why deterministic 1D -> 2D expansion is still rank-limited.

This script is intentionally independent of the project training pipeline. It
uses only the Python standard library and writes an SVG that can be opened
directly in a browser on a local machine.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 1D Gaussian expansion into a 2D target basin")
    parser.add_argument("--output", default="results/gaussian_1d_to_2d.svg")
    parser.add_argument("--csv", default="results/gaussian_1d_to_2d.csv")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=520)
    return parser.parse_args()


def randn():
    # Box-Muller, avoiding numpy so the script stays portable.
    u1 = max(random.random(), 1e-12)
    u2 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def target_2d():
    # Correlated anisotropic 2D Gaussian: a "true" 2D basin.
    x = randn()
    y = randn()
    return 1.7 * x + 0.4 * y, 0.35 * x + 0.9 * y


def linear_1d_to_2d():
    # A matrix W with shape [2, 1]. It expands the coordinate count, but the
    # sample covariance is still rank 1 because there is only one random source.
    z = randn()
    return 1.8 * z, 0.55 * z


def curve_1d_to_2d():
    # A nonlinear 1D curve in 2D. It bends, but still does not fill an area.
    z = randn()
    return 1.25 * z, 0.45 * (z * z - 1.0)


def noisy_1d_to_2d(noise_scale=0.25):
    # Add independent 2D jitter after the 1D map. This starts filling area, but
    # the added noise must carry the missing local degrees of freedom.
    x, y = linear_1d_to_2d()
    return x + noise_scale * randn(), y + noise_scale * randn()


def covariance(points):
    n = max(len(points), 1)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    cxx = sum((p[0] - mx) ** 2 for p in points) / n
    cyy = sum((p[1] - my) ** 2 for p in points) / n
    cxy = sum((p[0] - mx) * (p[1] - my) for p in points) / n
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = max(trace * trace - 4.0 * det, 0.0)
    eig1 = 0.5 * (trace + math.sqrt(disc))
    eig2 = 0.5 * (trace - math.sqrt(disc))
    rank_ratio = eig2 / max(eig1, 1e-12)
    return eig1, eig2, rank_ratio


def scale_points(groups, width, height):
    margin = 54
    all_points = [point for points in groups.values() for point in points]
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)
    pad_x = 0.12 * max(max_x - min_x, 1e-6)
    pad_y = 0.12 * max(max_y - min_y, 1e-6)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    def tx(point):
        x, y = point
        sx = margin + (x - min_x) / max(max_x - min_x, 1e-6) * (width - 2 * margin)
        sy = height - margin - (y - min_y) / max(max_y - min_y, 1e-6) * (height - 2 * margin)
        return sx, sy

    return tx


def svg_circle(x, y, r, fill, opacity=0.55):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" fill-opacity="{opacity}"/>'


def write_svg(path, groups, stats, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    tx = scale_points(groups, width, height)
    colors = {
        "target_2d": "#111827",
        "linear_1d_to_2d": "#2563eb",
        "curve_1d_to_2d": "#d97706",
        "noisy_1d_to_2d": "#059669",
    }
    labels = {
        "target_2d": "true 2D target Gaussian",
        "linear_1d_to_2d": "linear matrix W: 1D -> 2D",
        "curve_1d_to_2d": "nonlinear 1D curve in 2D",
        "noisy_1d_to_2d": "1D -> 2D plus independent jitter",
    }

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="54" y="34" font-family="Arial" font-size="22" fill="#111827">1D Gaussian expanded to 2D versus a true 2D target</text>',
        '<text x="54" y="58" font-family="Arial" font-size="13" fill="#4b5563">A matrix can increase coordinate count, but it cannot create missing independent dimensions by itself.</text>',
    ]

    for x in range(100, width, 100):
        lines.append(f'<line x1="{x}" y1="78" x2="{x}" y2="{height - 54}" stroke="#e5e7eb" stroke-width="1"/>')
    for y in range(100, height - 50, 80):
        lines.append(f'<line x1="54" y1="{y}" x2="{width - 54}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')

    for name, points in groups.items():
        for point in points:
            x, y = tx(point)
            lines.append(svg_circle(x, y, 2.5, colors[name], opacity=0.42 if name != "target_2d" else 0.34))

    legend_x = width - 350
    legend_y = 86
    lines.append(f'<rect x="{legend_x - 18}" y="{legend_y - 28}" width="330" height="154" rx="6" fill="white" stroke="#d1d5db"/>')
    for idx, name in enumerate(groups):
        y = legend_y + idx * 34
        eig1, eig2, rank_ratio = stats[name]
        lines.append(svg_circle(legend_x, y - 5, 5, colors[name], opacity=0.85))
        lines.append(
            f'<text x="{legend_x + 14}" y="{y}" font-family="Arial" font-size="13" fill="#111827">'
            f'{labels[name]} | eig2/eig1={rank_ratio:.3f}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, groups):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "x", "y"])
        writer.writeheader()
        for name, points in groups.items():
            for x, y in points:
                writer.writerow({"group": name, "x": x, "y": y})


def main():
    args = parse_args()
    random.seed(args.seed)
    groups = {
        "target_2d": [target_2d() for _ in range(args.samples)],
        "linear_1d_to_2d": [linear_1d_to_2d() for _ in range(args.samples)],
        "curve_1d_to_2d": [curve_1d_to_2d() for _ in range(args.samples)],
        "noisy_1d_to_2d": [noisy_1d_to_2d() for _ in range(args.samples)],
    }
    stats = {name: covariance(points) for name, points in groups.items()}
    write_svg(Path(args.output), groups, stats, args.width, args.height)
    write_csv(Path(args.csv), groups)
    print(f"wrote {args.output}")
    print(f"wrote {args.csv}")
    for name, (_eig1, _eig2, ratio) in stats.items():
        print(f"{name}: covariance eig2/eig1={ratio:.4f}")


if __name__ == "__main__":
    main()
