#!/usr/bin/env python3
"""Merge simple binary little-endian mesh PLY tiles into one PLY."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def read_mesh_ply(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    data = path.read_bytes()
    header, body = data.split(b"end_header\n", 1)
    text = header.decode("ascii", errors="strict")
    vertex_count = face_count = None
    for line in text.splitlines():
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face "):
            face_count = int(line.split()[-1])
    if vertex_count is None or face_count is None:
        raise ValueError(f"Unsupported PLY header: {path}")
    vertices = np.frombuffer(body[: vertex_count * 12], dtype="<f4").reshape(vertex_count, 3).copy()
    offset = vertex_count * 12
    faces: list[list[int]] = []
    for _ in range(face_count):
        n = body[offset]
        offset += 1
        face = list(struct.unpack_from("<" + "i" * n, body, offset))
        offset += 4 * n
        faces.append(face)
    return vertices, faces


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: list[list[int]]) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.astype("<f4", copy=False).tobytes())
        for face in faces:
            f.write(struct.pack("<B", len(face)))
            f.write(struct.pack("<" + "i" * len(face), *face))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    all_vertices = []
    all_faces: list[list[int]] = []
    vertex_offset = 0
    for path in sorted(args.input_dir.glob("*.ply")):
        vertices, faces = read_mesh_ply(path)
        all_vertices.append(vertices)
        all_faces.extend([[index + vertex_offset for index in face] for face in faces])
        vertex_offset += len(vertices)
        print(f"[merge_mesh_ply] {path.name}: {len(vertices)} verts, {len(faces)} faces")
    merged_vertices = np.concatenate(all_vertices, axis=0)
    write_mesh_ply(args.output, merged_vertices, all_faces)
    print(f"[merge_mesh_ply] wrote {args.output}: {len(merged_vertices)} verts, {len(all_faces)} faces")
    print(f"[merge_mesh_ply] xyz min={merged_vertices.min(axis=0).tolist()} max={merged_vertices.max(axis=0).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
