#!/usr/bin/env bash
# =============================================================================
# run.sh — Prepare textures and render a lunar surface image.
#
# Usage:
#   bash run.sh [--config config.json]
#
# All parameters are read from config.json.
# Edit config.json to change camera position, sun, render resolution, etc.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo " Lunar Surface Renderer"
echo " Config: ${CONFIG}"
echo "========================================"

echo ""
echo "[1/2] Preparing textures ..."
python3 "${SCRIPT_DIR}/prepare_textures.py" --config "${CONFIG}"

echo ""
echo "[2/2] Running Blender render ..."
blender --background --python "${SCRIPT_DIR}/lunar_render.py" -- --config "${CONFIG}"

OUTPUT_DIR="$(python3 -c "import json; print(json.load(open('${CONFIG}'))['paths']['output_dir'])")"
echo ""
echo "========================================"
echo " All done!"
echo "  Image  : ${OUTPUT_DIR}/lunar_render.png"
echo "  LatLon : ${OUTPUT_DIR}/lunar_render_latlon.json + .npz"
echo "========================================"
