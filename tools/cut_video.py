#!/usr/bin/env python3
"""
Cut a segment from a video between start and end time (in seconds).

Usage:
    python cut_video.py input.mp4 -s 10 -e 25 -o output.mp4
"""

import argparse
from moviepy.editor import VideoFileClip


def cut_video_segment(input_path, output_path, start_time, end_time):
    if start_time < 0:
        raise ValueError("start_time must be >= 0.")
    if end_time is not None and end_time <= start_time:
        raise ValueError("end_time must be greater than start_time.")

    # Load video
    clip = VideoFileClip(input_path)

    # If no end_time provided, cut until the end
    if end_time is None or end_time > clip.duration:
        end_time = clip.duration

    sub = clip.subclip(start_time, end_time)

    # Use original fps to avoid warnings
    fps = clip.fps

    # Write output
    sub.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        threads=4,
    )

    # Close clips
    sub.close()
    clip.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cut a segment from a video between start and end time (seconds)."
    )
    parser.add_argument("input", help="Input video file path.")
    parser.add_argument(
        "-s",
        "--start",
        type=float,
        required=True,
        help="Start time in seconds (e.g., 3.5).",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=float,
        required=False,
        help="End time in seconds (e.g., 12.0). If omitted, cuts until the end.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output video file path (e.g., clip.mp4).",
    )

    args = parser.parse_args()
    cut_video_segment(args.input, args.output, args.start, args.end)


if __name__ == "__main__":
    main()