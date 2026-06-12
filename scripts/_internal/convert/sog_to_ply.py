#!/usr/bin/env python3
"""Convert PlayCanvas/SuperSplat SOG gaussian files to standard 3DGS PLY."""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


SH_C0 = 0.28209479177387814
INV_SQRT2 = 1.0 / math.sqrt(2.0)


PLY_FIELDS = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)


def _read_sog_member(sog_path: Path, name: str) -> bytes:
    with zipfile.ZipFile(sog_path) as archive:
        return archive.read(name)


def _read_json(sog_path: Path) -> dict:
    return json.loads(_read_sog_member(sog_path, "meta.json"))


def _read_image(sog_path: Path, name: str) -> np.ndarray:
    data = _read_sog_member(sog_path, name)
    image = Image.open(io.BytesIO(data))
    return np.asarray(image)


def _valid_pixels(arr: np.ndarray, count: int) -> np.ndarray:
    return arr.reshape(-1, arr.shape[-1])[:count]


def _signed_log_decode(encoded: np.ndarray) -> np.ndarray:
    return np.sign(encoded) * (np.exp(np.abs(encoded)) - 1.0)


def _decode_means(sog_path: Path, meta: dict) -> np.ndarray:
    lower = _valid_pixels(_read_image(sog_path, meta["means"]["files"][0]), meta["count"]).astype(np.uint32)
    upper = _valid_pixels(_read_image(sog_path, meta["means"]["files"][1]), meta["count"]).astype(np.uint32)
    uint16 = lower + upper * 256
    mins = np.asarray(meta["means"]["mins"], dtype=np.float32)
    maxs = np.asarray(meta["means"]["maxs"], dtype=np.float32)
    logged = mins + (maxs - mins) * (uint16.astype(np.float32) / 65535.0)
    return _signed_log_decode(logged)


def _decode_codebook_rgb(sog_path: Path, section: dict) -> np.ndarray:
    pixels = _valid_pixels(_read_image(sog_path, section["files"][0]), section.get("count", 10**18))
    codebook = np.asarray(section["codebook"], dtype=np.float32)
    return codebook[pixels[:, :3].astype(np.int64)]


def _decode_sh0_and_opacity(sog_path: Path, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    rgba = _valid_pixels(_read_image(sog_path, meta["sh0"]["files"][0]), meta["count"])
    codebook = np.asarray(meta["sh0"]["codebook"], dtype=np.float32)
    f_dc = codebook[rgba[:, :3].astype(np.int64)]
    alpha = rgba[:, 3].astype(np.float32) / 255.0
    alpha = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    opacity = np.log(alpha / (1.0 - alpha)).reshape(-1, 1).astype(np.float32)
    return f_dc.astype(np.float32), opacity


def _decode_quats(sog_path: Path, meta: dict) -> np.ndarray:
    rgba = _valid_pixels(_read_image(sog_path, meta["quats"]["files"][0]), meta["count"]).astype(np.float32)
    dropped = np.clip(rgba[:, 3].astype(np.int32) - 252, 0, 3)
    stored = (rgba[:, :3] / 255.0) * (2.0 * INV_SQRT2) - INV_SQRT2
    quats = np.zeros((rgba.shape[0], 4), dtype=np.float32)
    src_idx = np.zeros(rgba.shape[0], dtype=np.int32)
    for component in range(4):
        mask = dropped != component
        quats[mask, component] = stored[mask, src_idx[mask]]
        src_idx[mask] += 1
    missing_sq = 1.0 - np.sum(quats * quats, axis=1)
    quats[np.arange(quats.shape[0]), dropped] = np.sqrt(np.clip(missing_sq, 0.0, 1.0))
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    return quats / np.clip(norms, 1e-8, None)


def decode_sog(sog_path: Path) -> np.ndarray:
    meta = _read_json(sog_path)
    count = int(meta["count"])
    xyz = _decode_means(sog_path, meta)
    scales = _decode_codebook_rgb(sog_path, meta["scales"])[:count].astype(np.float32)
    f_dc, opacity = _decode_sh0_and_opacity(sog_path, meta)
    quats = _decode_quats(sog_path, meta)
    normals = np.zeros_like(xyz, dtype=np.float32)
    f_rest = np.zeros((count, 45), dtype=np.float32)
    values = np.concatenate(
        [xyz.astype(np.float32), normals, f_dc, f_rest, opacity, scales, quats.astype(np.float32)], axis=1
    )
    return values


def _collect_lcc2_ranges(lcc2_path: Path, lod_depth: int | None) -> tuple[Path, list[str], list[tuple[int, int, int, str]]]:
    manifest = json.loads(lcc2_path.read_text(encoding="utf-8"))
    root = manifest["root"]
    base_dir = lcc2_path.parent
    splat_files = root["splatFiles"]
    ranges: list[tuple[int, int, int, str]] = []

    def walk(node: dict, depth: int) -> None:
        data = node.get("data") or {}
        if "3dgs" in data and (lod_depth is None or depth == lod_depth):
            splat = data["3dgs"]
            ranges.append((int(splat["name"]), int(splat["start"]), int(splat["count"]), str(node.get("id", ""))))
        for child in (node.get("child") or {}).values():
            walk(child, depth + 1)

    walk(root, 0)
    if not ranges:
        raise SystemExit(f"No 3DGS ranges found in {lcc2_path} for lod depth {lod_depth}")
    return base_dir, splat_files, ranges


def decode_lcc2(lcc2_path: Path, lod_depth: int | None) -> np.ndarray:
    base_dir, splat_files, ranges = _collect_lcc2_ranges(lcc2_path, lod_depth)
    by_file: dict[int, list[tuple[int, int, str]]] = {}
    for file_index, start, count, node_id in ranges:
        by_file.setdefault(file_index, []).append((start, count, node_id))

    chunks = []
    total = 0
    for file_index in sorted(by_file):
        rel_path = splat_files[file_index]
        if Path(rel_path).name == "env.sog":
            continue
        sog_path = base_dir / rel_path
        decoded = decode_sog(sog_path)
        file_total = 0
        for start, count, node_id in sorted(by_file[file_index]):
            end = start + count
            if end > decoded.shape[0]:
                raise SystemExit(
                    f"LCC2 range exceeds {sog_path.name}: node={node_id} start={start} count={count}"
                )
            chunks.append(decoded[start:end])
            file_total += count
        total += file_total
        print(f"[sog_to_ply] {sog_path.name}: selected {file_total} / {decoded.shape[0]} gaussians")

    if not chunks:
        raise SystemExit(f"No non-env SOG ranges selected from {lcc2_path}")
    values = np.concatenate(chunks, axis=0)
    print(f"[sog_to_ply] selected {total} gaussians from {len(ranges)} lcc2 ranges")
    return values


def write_binary_ply(path: Path, values: np.ndarray) -> None:
    dtype = np.dtype([(name, "<f4") for name in PLY_FIELDS])
    records = np.empty(values.shape[0], dtype=dtype)
    for index, name in enumerate(PLY_FIELDS):
        records[name] = values[:, index]
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {values.shape[0]}\n"
    header += "".join(f"property float {name}\n" for name in PLY_FIELDS)
    header += "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(records.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="SOG file or directory containing SOG files")
    parser.add_argument("output", type=Path, help="Output binary 3DGS PLY path")
    parser.add_argument("--include-env", action="store_true", help="Include env.sog if present")
    parser.add_argument("--lcc2", type=Path, help="LCC2 manifest to select SOG ranges instead of concatenating files")
    parser.add_argument(
        "--lod-depth",
        type=int,
        help="Tree depth to export from --lcc2. For this LCC2, depth 4 is the highest-detail layer.",
    )
    args = parser.parse_args()

    if args.lcc2:
        values = decode_lcc2(args.lcc2, args.lod_depth)
        total = values.shape[0]
    elif args.input.is_dir():
        sog_files = sorted(args.input.glob("*.sog"))
        if not args.include_env:
            sog_files = [p for p in sog_files if p.name != "env.sog"]

        if not sog_files:
            raise SystemExit(f"No SOG files found in {args.input}")

        chunks = []
        total = 0
        for sog_path in sog_files:
            chunk = decode_sog(sog_path)
            chunks.append(chunk)
            total += chunk.shape[0]
            print(f"[sog_to_ply] {sog_path.name}: {chunk.shape[0]} gaussians")

        values = np.concatenate(chunks, axis=0)
    else:
        values = decode_sog(args.input)
        total = values.shape[0]
        print(f"[sog_to_ply] {args.input.name}: {total} gaussians")

    write_binary_ply(args.output, values)
    mins = values[:, :3].min(axis=0)
    maxs = values[:, :3].max(axis=0)
    print(f"[sog_to_ply] wrote {args.output}: {total} gaussians")
    print(f"[sog_to_ply] xyz min={mins.tolist()} max={maxs.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
