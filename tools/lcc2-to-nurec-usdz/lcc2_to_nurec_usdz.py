#!/usr/bin/env python3
"""Convert XGRIDS LCC2 streamed SOG scenes to Isaac/Omniverse NuRec USDZ.

The pipeline is intentionally explicit:
1. Read the LCC2 manifest.
2. Select one LOD tree depth instead of concatenating all LODs.
3. Decode SOG V2 chunks to standard 3DGS PLY.
4. Merge mesh PLY tiles for optional collision/proxy geometry.
5. Call 3DGRUT to write NuRec USDZ.
6. Optionally add the merged mesh as invisible collision geometry.

Only steps 1-4 are implemented locally. Steps 5-6 use NVIDIA 3DGRUT's
official export scripts via a Python interpreter that has `threedgrut`
installed.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import struct
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


INV_SQRT2 = 1.0 / math.sqrt(2.0)
PLY_FIELDS = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)


@dataclass
class Lcc2Range:
    file_index: int
    start: int
    count: int
    node_id: str
    depth: int


@dataclass
class ArtifactSummary:
    input_root: str
    manifest: str
    output_dir: str
    scene_name: str
    selected_depth: int | str
    selected_gaussians: int
    selected_ranges: int
    gaussian_bbox_min: list[float]
    gaussian_bbox_max: list[float]
    gaussian_ply: str
    mesh_ply: str | None
    nurec_usdz: str | None
    collision_usdz: str | None
    validation: dict[str, Any]


def info(message: str) -> None:
    print(f"[lcc2-to-nurec] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_manifest(input_root: Path, manifest: Path | None) -> Path:
    if manifest:
        return manifest.resolve()
    candidates = sorted(input_root.rglob("*.lcc2"))
    if not candidates:
        raise SystemExit(f"No .lcc2 manifest found under {input_root}")
    if len(candidates) > 1:
        info(f"Multiple manifests found; using {candidates[0]}")
    return candidates[0].resolve()


def signed_log_decode(encoded: np.ndarray) -> np.ndarray:
    """Inverse of sign(x) * log(abs(x) + 1), as used by SOG V2 means."""
    return np.sign(encoded) * (np.exp(np.abs(encoded)) - 1.0)


def read_sog_member(sog_path: Path, name: str) -> bytes:
    with zipfile.ZipFile(sog_path) as archive:
        return archive.read(name)


def read_sog_json(sog_path: Path) -> dict[str, Any]:
    return json.loads(read_sog_member(sog_path, "meta.json"))


def read_sog_image(sog_path: Path, name: str) -> np.ndarray:
    data = read_sog_member(sog_path, name)
    return np.asarray(Image.open(io.BytesIO(data)))


def valid_pixels(arr: np.ndarray, count: int) -> np.ndarray:
    return arr.reshape(-1, arr.shape[-1])[:count]


def decode_means(sog_path: Path, meta: dict[str, Any]) -> np.ndarray:
    """Decode SOG V2 means.

    Important: meta.means.mins/maxs are bounds in log-transform space. The
    correct order is uint16 -> linear interpolate in log-space -> inverse log.
    Applying inverse log before the mins/maxs mapping compresses large LCC2
    scenes into a small room-sized blob.
    """
    lower = valid_pixels(read_sog_image(sog_path, meta["means"]["files"][0]), meta["count"]).astype(np.uint32)
    upper = valid_pixels(read_sog_image(sog_path, meta["means"]["files"][1]), meta["count"]).astype(np.uint32)
    uint16 = lower + upper * 256
    mins = np.asarray(meta["means"]["mins"], dtype=np.float32)
    maxs = np.asarray(meta["means"]["maxs"], dtype=np.float32)
    logged = mins + (maxs - mins) * (uint16.astype(np.float32) / 65535.0)
    return signed_log_decode(logged).astype(np.float32)


def decode_codebook_rgb(sog_path: Path, section: dict[str, Any], count: int) -> np.ndarray:
    pixels = valid_pixels(read_sog_image(sog_path, section["files"][0]), count)
    codebook = np.asarray(section["codebook"], dtype=np.float32)
    return codebook[pixels[:, :3].astype(np.int64)]


def decode_sh0_and_opacity(sog_path: Path, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgba = valid_pixels(read_sog_image(sog_path, meta["sh0"]["files"][0]), meta["count"])
    codebook = np.asarray(meta["sh0"]["codebook"], dtype=np.float32)
    f_dc = codebook[rgba[:, :3].astype(np.int64)].astype(np.float32)
    alpha = rgba[:, 3].astype(np.float32) / 255.0
    alpha = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    opacity = np.log(alpha / (1.0 - alpha)).reshape(-1, 1).astype(np.float32)
    return f_dc, opacity


def decode_quats(sog_path: Path, meta: dict[str, Any]) -> np.ndarray:
    rgba = valid_pixels(read_sog_image(sog_path, meta["quats"]["files"][0]), meta["count"]).astype(np.float32)
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
    meta = read_sog_json(sog_path)
    if meta.get("version") != 2:
        raise ValueError(f"Only SOG V2 is supported by this tool: {sog_path}")

    count = int(meta["count"])
    xyz = decode_means(sog_path, meta)
    scales = decode_codebook_rgb(sog_path, meta["scales"], count).astype(np.float32)
    f_dc, opacity = decode_sh0_and_opacity(sog_path, meta)
    quats = decode_quats(sog_path, meta).astype(np.float32)
    normals = np.zeros_like(xyz, dtype=np.float32)
    f_rest = np.zeros((count, 45), dtype=np.float32)
    return np.concatenate([xyz, normals, f_dc, f_rest, opacity, scales, quats], axis=1)


def collect_lcc2_ranges(manifest_obj: dict[str, Any], depth: int | None) -> list[Lcc2Range]:
    ranges: list[Lcc2Range] = []

    def walk(node: dict[str, Any], current_depth: int) -> None:
        data = node.get("data") or {}
        if "3dgs" in data and (depth is None or current_depth == depth):
            splat = data["3dgs"]
            ranges.append(
                Lcc2Range(
                    file_index=int(splat["name"]),
                    start=int(splat["start"]),
                    count=int(splat["count"]),
                    node_id=str(node.get("id", "")),
                    depth=current_depth,
                )
            )
        for child in (node.get("child") or {}).values():
            walk(child, current_depth + 1)

    walk(manifest_obj["root"], 0)
    if not ranges:
        raise SystemExit(f"No 3DGS ranges found for depth={depth}")
    return ranges


def count_ranges_by_depth(manifest_obj: dict[str, Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for item in collect_lcc2_ranges(manifest_obj, None):
        counts[item.depth] = counts.get(item.depth, 0) + item.count
    return dict(sorted(counts.items()))


def resolve_lod_depth(manifest_obj: dict[str, Any], requested: str) -> int:
    counts_by_depth = count_ranges_by_depth(manifest_obj)
    if requested == "highest":
        return max(counts_by_depth)
    if requested == "lowest":
        return min(counts_by_depth)
    try:
        depth = int(requested)
    except ValueError as exc:
        raise SystemExit(f"Invalid --lod-depth {requested!r}; use highest, lowest, or an integer") from exc
    if depth not in counts_by_depth:
        raise SystemExit(f"LOD depth {depth} not present. Available depths: {sorted(counts_by_depth)}")
    return depth


def decode_lcc2_to_values(manifest_path: Path, lod_depth: int, include_env: bool = False) -> tuple[np.ndarray, int]:
    manifest_obj = read_json(manifest_path)
    root = manifest_obj["root"]
    base_dir = manifest_path.parent
    splat_files = root["splatFiles"]
    ranges = collect_lcc2_ranges(manifest_obj, lod_depth)

    by_file: dict[int, list[Lcc2Range]] = {}
    for item in ranges:
        by_file.setdefault(item.file_index, []).append(item)

    chunks = []
    selected_ranges = 0
    for file_index in sorted(by_file):
        rel_path = splat_files[file_index]
        if Path(rel_path).name == "env.sog" and not include_env:
            continue
        sog_path = base_dir / rel_path
        decoded = decode_sog(sog_path)
        file_count = 0
        for item in sorted(by_file[file_index], key=lambda value: value.start):
            end = item.start + item.count
            if end > decoded.shape[0]:
                raise SystemExit(
                    f"LCC2 range exceeds {sog_path.name}: node={item.node_id} start={item.start} count={item.count}"
                )
            chunks.append(decoded[item.start : end])
            file_count += item.count
            selected_ranges += 1
        info(f"{sog_path.name}: selected {file_count} / {decoded.shape[0]} gaussians")

    if not chunks:
        raise SystemExit(f"No non-env SOG ranges selected from {manifest_path}")

    values = np.concatenate(chunks, axis=0)
    return values, selected_ranges


def write_gaussian_ply(path: Path, values: np.ndarray) -> None:
    dtype = np.dtype([(name, "<f4") for name in PLY_FIELDS])
    records = np.empty(values.shape[0], dtype=dtype)
    for index, name in enumerate(PLY_FIELDS):
        records[name] = values[:, index]

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {values.shape[0]}\n"
    header += "".join(f"property float {name}\n" for name in PLY_FIELDS)
    header += "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        file.write(records.tobytes())


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
        raise ValueError(f"Unsupported mesh PLY header: {path}")
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
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        file.write(vertices.astype("<f4", copy=False).tobytes())
        for face in faces:
            file.write(struct.pack("<B", len(face)))
            file.write(struct.pack("<" + "i" * len(face), *face))


def merge_mesh_tiles(mesh_dir: Path, output: Path) -> tuple[int, int, list[float], list[float]]:
    mesh_files = sorted(mesh_dir.glob("*.ply"))
    if not mesh_files:
        raise SystemExit(f"No mesh PLY files found in {mesh_dir}")

    all_vertices = []
    all_faces: list[list[int]] = []
    vertex_offset = 0
    for path in mesh_files:
        vertices, faces = read_mesh_ply(path)
        all_vertices.append(vertices)
        all_faces.extend([[index + vertex_offset for index in face] for face in faces])
        vertex_offset += len(vertices)
        info(f"{path.name}: {len(vertices)} mesh vertices, {len(faces)} mesh faces")

    merged_vertices = np.concatenate(all_vertices, axis=0)
    write_mesh_ply(output, merged_vertices, all_faces)
    return (
        int(len(merged_vertices)),
        int(len(all_faces)),
        merged_vertices.min(axis=0).tolist(),
        merged_vertices.max(axis=0).tolist(),
    )


def run_command(command: list[str], cwd: Path | None = None) -> None:
    info("running: " + " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def require_module(python: str, module_name: str) -> None:
    command = [python, "-c", f"import {module_name}; print({module_name!r}, 'ok')"]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Python interpreter {python!r} cannot import {module_name!r}.\n"
            f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc


def transcode_to_nurec(threedgrut_python: str, gaussian_ply: Path, output_usdz: Path) -> None:
    require_module(threedgrut_python, "threedgrut")
    run_command(
        [
            threedgrut_python,
            "-m",
            "threedgrut.export.scripts.transcode",
            str(gaussian_ply),
            "-o",
            str(output_usdz),
            "--format",
            "nurec",
        ]
    )


def add_collision_mesh(threedgrut_python: str, input_usdz: Path, output_usdz: Path, mesh_ply: Path) -> None:
    require_module(threedgrut_python, "pxr")
    run_command(
        [
            threedgrut_python,
            "-m",
            "threedgrut.export.scripts.add_mesh_to_usdz",
            "--input_usdz",
            str(input_usdz),
            "--output_usdz",
            str(output_usdz),
            "--mesh_ply",
            str(mesh_ply),
            "--set_collision",
            "--set_invisible",
        ]
    )


def validate_usdz(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.exists(),
        "path": str(path),
        "files": [],
        "has_nurec_payload": False,
        "has_nurec_volume_marker": False,
        "crop_min": None,
        "crop_max": None,
        "has_mesh_usd": False,
    }
    if not path.exists():
        return result

    with zipfile.ZipFile(path) as archive:
        result["files"] = [{"name": item.filename, "size": item.file_size} for item in archive.infolist()]
        result["has_nurec_payload"] = any(item.filename.endswith(".nurec") for item in archive.infolist())
        result["has_mesh_usd"] = any(item.filename == "mesh.usd" for item in archive.infolist())
        usd_text = "\n".join(
            archive.read(item.filename).decode("utf-8", errors="ignore")
            for item in archive.infolist()
            if item.filename.endswith((".usd", ".usda"))
        )

    result["has_nurec_volume_marker"] = "omni:nurec:isNuRecVolume" in usd_text
    for label, key in [("crop:minBounds", "crop_min"), ("crop:maxBounds", "crop_max")]:
        marker = f"omni:nurec:{label} = "
        if marker in usd_text:
            tail = usd_text.split(marker, 1)[1].splitlines()[0].strip()
            result[key] = tail
    return result


def scene_stem(manifest_obj: dict[str, Any], manifest_path: Path) -> str:
    raw = str(manifest_obj.get("guid") or manifest_path.stem)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    return safe.strip("_") or "lcc2_scene"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an XGRIDS LCC2 streamed SOG scene to 3DGS PLY and optional 3DGRUT NuRec USDZ."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing the .lcc2 manifest and data/")
    parser.add_argument("--manifest", type=Path, help="Explicit .lcc2 manifest path. Auto-discovered if omitted.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated PLY/USDZ artifacts.")
    parser.add_argument(
        "--lod-depth",
        default="highest",
        help="LOD tree depth to export: highest, lowest, or an integer. Default: highest.",
    )
    parser.add_argument("--include-env", action="store_true", help="Include env.sog if selected by the manifest.")
    parser.add_argument("--skip-mesh", action="store_true", help="Do not merge data/mesh/*.ply.")
    parser.add_argument("--skip-usdz", action="store_true", help="Only write PLY artifacts; do not call 3DGRUT.")
    parser.add_argument(
        "--threedgrut-python",
        default=os.environ.get("THREEDGRUT_PYTHON", sys.executable),
        help="Python interpreter with threedgrut/usd-core installed. Default: THREEDGRUT_PYTHON or current Python.",
    )
    parser.add_argument(
        "--collision",
        action="store_true",
        help="After NuRec export, add merged mesh as invisible collision/proxy geometry.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path. Defaults to output-dir/summary.json.")
    return parser.parse_args()


def ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file without --force: {path}")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = discover_manifest(input_root, args.manifest)
    manifest_obj = read_json(manifest_path)
    selected_depth = resolve_lod_depth(manifest_obj, args.lod_depth)
    counts_by_depth = count_ranges_by_depth(manifest_obj)

    info(f"manifest: {manifest_path}")
    info(f"LOD counts by tree depth: {counts_by_depth}")
    info(f"selected depth: {selected_depth}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = scene_stem(manifest_obj, manifest_path)
    gaussian_ply = output_dir / f"{stem}_lod{selected_depth}_gaussians.ply"
    mesh_ply = output_dir / f"{stem}_mesh.ply"
    nurec_usdz = output_dir / f"{stem}_lod{selected_depth}_nurec.usdz"
    collision_usdz = output_dir / f"{stem}_lod{selected_depth}_nurec_with_collision.usdz"
    summary_path = args.summary.resolve() if args.summary else output_dir / f"{stem}_summary.json"

    for path in [gaussian_ply, summary_path]:
        ensure_can_write(path, args.force)
    if not args.skip_mesh:
        ensure_can_write(mesh_ply, args.force)
    if not args.skip_usdz:
        ensure_can_write(nurec_usdz, args.force)
        if args.collision:
            ensure_can_write(collision_usdz, args.force)

    values, selected_ranges = decode_lcc2_to_values(manifest_path, selected_depth, include_env=args.include_env)
    bbox_min = values[:, :3].min(axis=0).tolist()
    bbox_max = values[:, :3].max(axis=0).tolist()
    write_gaussian_ply(gaussian_ply, values)
    info(f"wrote {gaussian_ply}: {values.shape[0]} gaussians")
    info(f"gaussian bbox min={bbox_min} max={bbox_max}")

    mesh_summary: dict[str, Any] | None = None
    mesh_output: Path | None = None
    if not args.skip_mesh:
        mesh_dir = manifest_path.parent / "data" / "mesh"
        verts, faces, mesh_min, mesh_max = merge_mesh_tiles(mesh_dir, mesh_ply)
        mesh_output = mesh_ply
        mesh_summary = {"vertices": verts, "faces": faces, "bbox_min": mesh_min, "bbox_max": mesh_max}
        info(f"wrote {mesh_ply}: {verts} vertices, {faces} faces")

    validation: dict[str, Any] = {}
    nurec_output: Path | None = None
    collision_output: Path | None = None
    if not args.skip_usdz:
        transcode_to_nurec(args.threedgrut_python, gaussian_ply, nurec_usdz)
        nurec_output = nurec_usdz
        validation["nurec_usdz"] = validate_usdz(nurec_usdz)
        if args.collision:
            if mesh_output is None:
                raise SystemExit("--collision requires mesh output; remove --skip-mesh")
            add_collision_mesh(args.threedgrut_python, nurec_usdz, collision_usdz, mesh_output)
            collision_output = collision_usdz
            validation["collision_usdz"] = validate_usdz(collision_usdz)

    summary = ArtifactSummary(
        input_root=str(input_root),
        manifest=str(manifest_path),
        output_dir=str(output_dir),
        scene_name=str(manifest_obj.get("name") or manifest_path.stem),
        selected_depth=selected_depth,
        selected_gaussians=int(values.shape[0]),
        selected_ranges=selected_ranges,
        gaussian_bbox_min=[float(v) for v in bbox_min],
        gaussian_bbox_max=[float(v) for v in bbox_max],
        gaussian_ply=str(gaussian_ply),
        mesh_ply=str(mesh_output) if mesh_output else None,
        nurec_usdz=str(nurec_output) if nurec_output else None,
        collision_usdz=str(collision_output) if collision_output else None,
        validation=validation,
    )
    payload = asdict(summary)
    payload["lod_counts_by_depth"] = counts_by_depth
    payload["mesh"] = mesh_summary
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    info(f"wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
