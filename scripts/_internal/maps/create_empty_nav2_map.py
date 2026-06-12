#!/usr/bin/env python3
"""Create a simple empty occupancy grid map for Nav2 demos."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an empty PGM/YAML map for Nav2.")
    parser.add_argument("--output_dir", type=Path, default=Path("maps/nav2_empty"))
    parser.add_argument("--name", type=str, default="empty_world")
    parser.add_argument("--size", type=float, default=20.0, help="Map width/height in meters.")
    parser.add_argument("--resolution", type=float, default=0.05, help="Map resolution in meters per pixel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pixels = int(round(args.size / args.resolution))
    origin = -0.5 * args.size

    pgm_path = args.output_dir / f"{args.name}.pgm"
    yaml_path = args.output_dir / f"{args.name}.yaml"

    row = " ".join(["254"] * pixels)
    with pgm_path.open("w", encoding="ascii") as f:
        f.write("P2\n")
        f.write(f"{pixels} {pixels}\n")
        f.write("255\n")
        for _ in range(pixels):
            f.write(row)
            f.write("\n")

    yaml_path.write_text(
        "\n".join(
            [
                f"image: {pgm_path.name}",
                "mode: trinary",
                f"resolution: {args.resolution}",
                f"origin: [{origin}, {origin}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(yaml_path.resolve())


if __name__ == "__main__":
    main()
