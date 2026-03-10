import argparse
import os
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Assemble frames/ PNGs into an MP4 video using ffmpeg.")
    p.add_argument("--batch", required=True,
                   help="Batch output directory (must contain a frames/ sub-folder)")
    p.add_argument("--fps", type=float, default=5.0,
                   help="Frames per second in the output video (default: 5)")
    p.add_argument("--out", default="timelapse.mp4",
                   help="Output filename, placed inside frames/ (default: timelapse.mp4)")
    p.add_argument("--crf", type=int, default=18,
                   help="H.264 CRF quality 0–51, lower = better (default: 18)")
    return p.parse_args()


def main():
    args = parse_args()

    batch_dir  = os.path.abspath(args.batch)
    frames_dir = os.path.join(batch_dir, "frames")

    if not os.path.isdir(frames_dir):
        sys.exit(f"[ERROR] frames/ directory not found: {frames_dir}")

    pngs = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".png"))
    if not pngs:
        sys.exit("[ERROR] No PNG files found in frames/")

    print(f"Found {len(pngs)} frame(s) at {args.fps} fps")
    print(f"  → estimated duration: {len(pngs) / args.fps:.1f} s")

    out_path  = os.path.join(frames_dir, args.out)
    list_path = os.path.join(frames_dir, "_concat_list.txt")

    # Write concat list next to the frames (avoids /tmp filesystem issues)
    with open(list_path, "w", encoding="utf-8") as tf:
        for fname in pngs:
            abs_png = os.path.join(frames_dir, fname)
            tf.write(f"file '{abs_png}'\n")
            tf.write(f"duration {1.0 / args.fps:.6f}\n")
        # repeat last frame so ffmpeg encodes it fully
        tf.write(f"file '{os.path.join(frames_dir, pngs[-1])}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        out_path,
    ]

    print(f"\nRunning ffmpeg …\n  {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("[ERROR] ffmpeg not found. Install with:  sudo apt install ffmpeg")
    except subprocess.CalledProcessError as e:
        sys.exit(f"[ERROR] ffmpeg failed (exit {e.returncode})")
    finally:
        if os.path.exists(list_path):
            os.unlink(list_path)

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"\n=== Done: {out_path}  ({size_mb:.1f} MB) ===")


if __name__ == "__main__":
    main()