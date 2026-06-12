#!/usr/bin/env python3
"""Create a static Nav2 map and Isaac obstacle scene for a substation room."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a substation-like Nav2 map and matching Isaac scene JSON.")
    parser.add_argument("--output_dir", type=Path, default=Path("maps/nav2_substation"))
    parser.add_argument("--name", type=str, default="substation_room")
    parser.add_argument("--resolution", type=float, default=0.05)
    return parser.parse_args()


def add_rect(
    obstacles: list[dict],
    name: str,
    x: float,
    y: float,
    sx: float,
    sy: float,
    height: float,
    kind: str,
    *,
    z: float | None = None,
    nav_occupied: bool = True,
) -> None:
    center_z = height * 0.5 if z is None else z
    obstacles.append(
        {
            "name": name,
            "kind": kind,
            "center": [x, y, center_z],
            "size": [sx, sy, height],
            "yaw": 0.0,
            "nav_occupied": nav_occupied,
        }
    )


def add_button(
    obstacles: list[dict],
    buttons: list[dict],
    name: str,
    cabinet_name: str,
    x: float,
    cabinet_y: float,
    face: str,
    *,
    z: float = 0.96,
) -> None:
    """Add a red button on a cabinet face plus its Nav2 stance pose."""

    cabinet_depth = 0.85
    button_depth = 0.40
    button_size = [0.20, button_depth, 0.20]
    approach_clearance = 0.55
    if face == "south":
        normal_y = -1.0
        center_y = cabinet_y - cabinet_depth * 0.5 - button_depth * 0.5
        approach_y = cabinet_y - cabinet_depth * 0.5 - approach_clearance
        yaw = 1.5708
    elif face == "north":
        normal_y = 1.0
        center_y = cabinet_y + cabinet_depth * 0.5 + button_depth * 0.5
        approach_y = cabinet_y + cabinet_depth * 0.5 + approach_clearance
        yaw = -1.5708
    else:
        raise ValueError(f"Unsupported button face: {face}")

    add_rect(
        obstacles,
        name,
        x,
        center_y,
        button_size[0],
        button_size[1],
        button_size[2],
        "button",
        z=z,
        nav_occupied=False,
    )
    buttons.append(
        {
            "name": name,
            "cabinet": cabinet_name,
            "face": face,
            "center": [x, center_y, z],
            "size": button_size,
            "normal": [0.0, normal_y, 0.0],
            "approach_goal": [x, approach_y, yaw],
            "press_pose": [x, center_y + normal_y * button_depth * 0.5, z],
        }
    )


def build_scene() -> dict:
    room = {
        "origin": [-2.0, -7.5],
        "size": [21.0, 15.0],
        "resolution": 0.05,
        "wall_height": 2.2,
        "cabinet_height": 1.8,
    }
    obstacles: list[dict] = []

    ox, oy = room["origin"]
    sx, sy = room["size"]
    wall = 0.20
    cx = ox + sx * 0.5
    cy = oy + sy * 0.5
    add_rect(obstacles, "wall_bottom", cx, oy + wall * 0.5, sx, wall, room["wall_height"], "wall")
    add_rect(obstacles, "wall_top", cx, oy + sy - wall * 0.5, sx, wall, room["wall_height"], "wall")
    add_rect(obstacles, "wall_left", ox + wall * 0.5, cy, wall, sy, room["wall_height"], "wall")
    add_rect(obstacles, "wall_right", ox + sx - wall * 0.5, cy, wall, sy, room["wall_height"], "wall")

    # Three rows of large switchgear cabinets. The gaps are intentional so Nav2
    # has multiple homotopy-like choices while the map stays easy to inspect.
    cabinet_h = room["cabinet_height"]
    row_ys = [-3.8, 0.0, 3.8]
    row_xs = [4.2, 8.6, 13.0, 16.4]
    cabinet_names: dict[tuple[int, int], str] = {}
    for row_index, y in enumerate(row_ys, start=1):
        for col_index, x in enumerate(row_xs, start=1):
            width_x = 2.2 if col_index < 4 else 1.6
            cabinet_name = f"cabinet_r{row_index}_c{col_index}"
            cabinet_names[(row_index, col_index)] = cabinet_name
            add_rect(
                obstacles,
                cabinet_name,
                x,
                y,
                width_x,
                0.85,
                cabinet_h,
                "cabinet",
            )

    # Red cabinet buttons for the operation demo. They are visible 3D targets
    # attached to cabinet faces, but intentionally do not enter the 2D Nav2 map.
    buttons: list[dict] = []
    button_specs = [
        (1, 1, "south"),
        (1, 2, "south"),
        (2, 2, "north"),
        (2, 4, "north"),
        (3, 2, "south"),
        (3, 3, "south"),
    ]
    for button_index, (row_index, col_index, face) in enumerate(button_specs, start=1):
        button_x = row_xs[col_index - 1]
        button_z = 0.96
        if button_index == 1:
            # 第一颗按钮用于明天的操作演示：放在柜面上更靠近 G1 右臂可达区的位置。
            # 这仍然是柜面真实按钮，只是避开了原点位对无手掌 G1 的侧向/下探极限。
            button_x -= 0.20
        add_button(
            obstacles,
            buttons,
            f"red_button_{button_index}",
            cabinet_names[(row_index, col_index)],
            button_x,
            row_ys[row_index - 1],
            face,
            z=button_z,
        )

    return {
        "description": "静态配电站房间：外墙 + 三排配电柜 + 柜面红色按钮操作点。坐标以机器人启动点为 map/odom 原点。",
        "coordinate_frame": "odom_at_start",
        "room": room,
        "button_demo": {
            "description": "红色按钮不参与 2D 占据图，只作为柜面可视操作目标；approach_goal 是底盘站位，press_pose 是后续手腕/工具末端按压目标。",
            "home_goal": [0.8, 0.0, 0.0],
            "buttons": buttons,
        },
        "obstacles": obstacles,
    }


def world_to_pixel(x: float, y: float, origin: list[float], resolution: float, height_px: int) -> tuple[int, int]:
    ix = int((x - origin[0]) / resolution)
    iy = int((y - origin[1]) / resolution)
    row = height_px - 1 - iy
    return ix, row


def paint_rect(grid: list[list[int]], origin: list[float], resolution: float, obstacle: dict, margin: float = 0.0) -> None:
    width_px = len(grid[0])
    height_px = len(grid)
    cx, cy, _ = obstacle["center"]
    sx, sy, _ = obstacle["size"]
    min_x = cx - sx * 0.5 - margin
    max_x = cx + sx * 0.5 + margin
    min_y = cy - sy * 0.5 - margin
    max_y = cy + sy * 0.5 + margin
    ix0, row1 = world_to_pixel(min_x, min_y, origin, resolution, height_px)
    ix1, row0 = world_to_pixel(max_x, max_y, origin, resolution, height_px)
    ix0 = max(0, min(width_px - 1, ix0))
    ix1 = max(0, min(width_px - 1, ix1))
    row0 = max(0, min(height_px - 1, row0))
    row1 = max(0, min(height_px - 1, row1))
    for row in range(min(row0, row1), max(row0, row1) + 1):
        for ix in range(min(ix0, ix1), max(ix0, ix1) + 1):
            grid[row][ix] = 0


def main() -> None:
    args = parse_args()
    scene = build_scene()
    scene["room"]["resolution"] = args.resolution
    args.output_dir.mkdir(parents=True, exist_ok=True)

    origin = scene["room"]["origin"]
    size = scene["room"]["size"]
    width_px = int(round(size[0] / args.resolution))
    height_px = int(round(size[1] / args.resolution))
    grid = [[254 for _ in range(width_px)] for _ in range(height_px)]

    for obstacle in scene["obstacles"]:
        if not obstacle.get("nav_occupied", True):
            continue
        paint_rect(grid, origin, args.resolution, obstacle)

    pgm_path = args.output_dir / f"{args.name}.pgm"
    yaml_path = args.output_dir / f"{args.name}.yaml"
    scene_path = args.output_dir / f"{args.name}_scene.json"

    with pgm_path.open("w", encoding="ascii") as f:
        f.write("P2\n")
        f.write(f"{width_px} {height_px}\n")
        f.write("255\n")
        for row in grid:
            f.write(" ".join(str(value) for value in row))
            f.write("\n")

    yaml_path.write_text(
        "\n".join(
            [
                f"image: {pgm_path.name}",
                "mode: trinary",
                f"resolution: {args.resolution}",
                f"origin: [{origin[0]}, {origin[1]}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )
    scene_path.write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")

    print(yaml_path.resolve())
    print(scene_path.resolve())


if __name__ == "__main__":
    main()
