#!/usr/bin/env python3
import os
import re
import cv2
import argparse
from typing import List, Tuple

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def extract_numeric_key(filename: str) -> Tuple[int, str]:
    """
    Returns a sort key that prioritizes the first integer found in the filename.
    If no integer exists, it falls back to the filename (lexicographic).
    """
    m = re.search(r'(\d+)', os.path.basename(filename))
    if m:
        return (int(m.group(1)), filename.lower())
    # place non-numeric names after numeric ones by using a large sentinel
    return (10**12, filename.lower())

def find_images(folder: str, recursive: bool) -> List[str]:
    paths = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and f.lower().endswith(IMAGE_EXTS):
                paths.append(p)
    return paths

def images_to_video(
    image_paths: List[str],
    out_path: str,
    fps: int = 5,
    width: int = 640,
    height: int = 360,
    codec: str = "mp4v",
    resize_to_first: bool = True
):
    if not image_paths:
        raise ValueError("No images found to create the video.")

    # Sort images by numeric key
    image_paths = sorted(image_paths, key=extract_numeric_key)
    image_paths = image_paths[::6]  # For testing, select every 2nd image

    # Read first image to get frame size
    first = cv2.imread(image_paths[0])
    if first is None:
        raise RuntimeError(f"Failed to read first image: {image_paths[0]}")
    # height, width = first.shape[:2]

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. "
                           "Try a different codec (e.g., 'avc1' or 'MJPG') or output extension (e.g., .avi).")

    # Write first frame
    writer.write(first)

    # Loop over remaining images
    for i, p in enumerate(image_paths[1:], start=2):
        frame = cv2.imread(p)
        if frame is None:
            print(f"[WARN] Skipping unreadable image: {p}")
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            if resize_to_first:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            else:
                print(f"[WARN] Skipping {p} due to size mismatch {frame.shape[1]}x{frame.shape[0]} "
                      f"(expected {width}x{height}).")
                continue
        writer.write(frame)

    writer.release()
    print(f"✅ Video saved to: {out_path}")
    print(f"Frames written: {len(image_paths)}  |  FPS: {fps}  |  Size: {width}x{height}")

def main():
    parser = argparse.ArgumentParser(
        description="Sort images by number in filename and convert to a 5 FPS video (one frame per image)."
    )
    parser.add_argument("--input", "-i", required=True, help="Folder containing images.")
    parser.add_argument("--output", "-o", default="output.mp4", help="Output video path (e.g., output.mp4).")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second (default: 5).")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively.")
    parser.add_argument("--codec", default="mp4v",
                        help="FourCC codec (e.g., mp4v, avc1, MJPG). Default: mp4v.")
    args = parser.parse_args()

    images = find_images(args.input, args.recursive)
    if not images:
        raise SystemExit(f"No images with extensions {IMAGE_EXTS} found in: {args.input}")

    images_to_video(images, args.output, fps=args.fps, codec=args.codec)

if __name__ == "__main__":
    main()