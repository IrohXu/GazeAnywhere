#!/usr/bin/env python3
"""
Merge multiple videos into one with 1-second crossfade transitions.

Usage:
    python merge_videos.py -o output.mp4 input1.mp4 input2.mp4 input3.mp4 ...
"""

import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips


def merge_videos_with_transition(input_paths, output_path, transition_duration=1.0):
    if len(input_paths) < 1:
        raise ValueError("You must provide at least one input video.")

    # Load all clips
    clips = [VideoFileClip(p) for p in input_paths]

    # Apply crossfade-in to all clips except the first
    faded_clips = [clips[0]]
    for clip in clips[1:]:
        faded_clips.append(clip.crossfadein(transition_duration))

    # Concatenate with negative padding so clips overlap by `transition_duration`
    final = concatenate_videoclips(
        faded_clips,
        method="compose",
        padding=-transition_duration,
    )

    # Use the fps of the first clip to avoid warnings
    fps = clips[0].fps

    # Write output video
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        threads=4,
    )

    # Close all clips to free resources
    final.close()
    for c in clips:
        c.close()


def main():
    parser = argparse.ArgumentParser(
        description="Merge videos with 1-second crossfade transitions."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input video files in the order you want them merged.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output video file path (e.g., merged.mp4).",
    )
    parser.add_argument(
        "-t",
        "--transition",
        type=float,
        default=1.0,
        help="Transition (crossfade) duration in seconds (default: 1.0).",
    )

    args = parser.parse_args()
    merge_videos_with_transition(args.inputs, args.output, args.transition)


if __name__ == "__main__":
    main()