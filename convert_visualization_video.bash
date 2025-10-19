#!/usr/bin/env bash
# batch_convert_visualizations.sh
# Convert many visualization folders into videos.
# Usage:
#   ./batch_convert_visualizations.sh /path/to/input_root /path/to/output_root
# Example:
#   ./batch_convert_visualizations.sh \
#     "/projects/illinois/eng/cs/jrehg/datasets-irb/devsci_autism/ChildGaze/visualization_child_large/visualization/data" \
#     "/projects/illinois/eng/cs/jrehg/users/xucao2/ChildGaze/output/video_output"

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <INPUT_DIR> <OUTPUT_DIR>"
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Make sure output dir exists
mkdir -p "$OUTPUT_DIR"

# Find immediate subdirectories in INPUT_DIR and convert each one
# (adjust -maxdepth if your structure is deeper)
find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' vis_dir; do
  base="$(basename "$vis_dir")"
  out_path="$OUTPUT_DIR/${base}.mp4"

  # Skip if already converted
  if [[ -f "$out_path" ]]; then
    echo "[SKIP] $out_path already exists."
    continue
  fi

  echo "[RUN ] Converting: $vis_dir -> $out_path"
  python tools/convert_visualization_to_video.py \
    --input  "$vis_dir" \
    --output "$out_path"
done

echo "All done."