#!/usr/bin/env bash
set -euo pipefail

USDZ_PATH="${1:-/workspace/data/substation_final_gs_mesh_collision.usdz}"
ISAACLAB_SH="${ISAACLAB_PATH:-/workspace/isaaclab}/isaaclab.sh"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPEN_STAGE_SCRIPT="${REPO_ROOT}/scripts/_internal/isaac/open_substation_stage.py"

if [[ ! -f "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] Isaac Lab launcher not found: ${ISAACLAB_SH}" >&2
    echo "Set ISAACLAB_PATH or edit this script to the correct Isaac Lab path." >&2
    exit 1
fi

if [[ ! -f "${USDZ_PATH}" ]]; then
    echo "[ERROR] USDZ file not found: ${USDZ_PATH}" >&2
    echo "Pass the USDZ path as the first argument, for example:" >&2
    echo "  $0 /workspace/data/substation_final_gs_mesh_collision.usdz" >&2
    exit 1
fi

export TERM="${TERM:-xterm}"
if [[ "${TERM}" == "dumb" ]]; then
    export TERM=xterm
fi

export SUBSTATION_USDZ_PATH="${USDZ_PATH}"

exec "${ISAACLAB_SH}" -s \
    --exec "${OPEN_STAGE_SCRIPT}" \
    --/app/window/width=2880 \
    --/app/window/height=1800
