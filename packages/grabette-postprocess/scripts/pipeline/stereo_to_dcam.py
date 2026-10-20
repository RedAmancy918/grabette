#!/usr/bin/env python3
"""Turn a stereo_uvc episode (raw side-by-side MJPG) into the usual dcam_* files.

The UVC stereo camera has no on-board depth, so the Pi only records
dcam_stereo.mjpeg + timestamps + calibration (see grabette/hardware/stereo_uvc.py).
This computes what the Gemini 305 would have produced on the device:

    dcam_left.mp4               H.264, rectified left eye, grayscale
    dcam_depth.mkv              uint16 mm, lossless FFV1, aligned to dcam_left
    dcam_left_timestamps.json   copied from dcam_stereo_timestamps.json
    dcam_depth_timestamps.json  (one depth frame per stereo frame)
    dcam_calib_offline.json     rectified fx/fy/cx/cy + baseline (no imu_to_cam)

after which convert_episode_to_oak.py / run_oak_slam.py run unchanged.

Rectification and SGBM come from stereokit, with the block size derived from
the eye width and the calibration's measured disparity_offset, as in
`stereokit view`. Run it from stereokit's environment:

    uv run --project /path/to/stereokit python scripts/pipeline/stereo_to_dcam.py -i EPISODE_DIR
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from stereokit import calibrate
from stereokit.depth import DepthEstimator
from stereokit.rectify import Rectifier


def spawn_ffmpeg(args: list[str], size: tuple[int, int], pix_fmt: str, fps: float, out: Path):
    """ffmpeg reading raw frames of `pix_fmt` on stdin."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{size[0]}x{size[1]}",
           "-r", f"{fps:.3f}", "-i", "-", *args, str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def convert(ep_dir: Path, max_depth: float) -> None:
    calib = calibrate.load(ep_dir / "dcam_calib_stereo.json")
    swap = json.loads((ep_dir / "dcam_calib.json").read_text())["swap_eyes"]
    samples = json.loads((ep_dir / "dcam_stereo_timestamps.json").read_text())["samples"]

    rect = Rectifier(calib)
    est = DepthEstimator(fx=rect.fx, baseline_m=rect.baseline_m,
                         image_width=rect.eye_size[0], max_depth=max_depth,
                         disparity_offset=calibrate.disparity_offset(calib))
    w, h = rect.eye_size
    # Nominal rate for the containers only; real timing is in the sidecars.
    span_s = (samples[-1]["host_ms"] - samples[0]["host_ms"]) / 1000.0
    fps = (len(samples) - 1) / span_s if span_s > 0 else 30.0

    left_enc = spawn_ffmpeg(["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                             "-pix_fmt", "yuv420p"],
                            (w, h), "gray", fps, ep_dir / "dcam_left.mp4")
    depth_enc = spawn_ffmpeg(["-c:v", "ffv1", "-level", "3"],
                             (w, h), "gray16le", fps, ep_dir / "dcam_depth.mkv")

    cap = cv2.VideoCapture(str(ep_dir / "dcam_stereo.mjpeg"), cv2.CAP_FFMPEG)
    n, valid = 0, 0.0
    t0 = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mid = frame.shape[1] // 2
        left, right = frame[:, :mid], frame[:, mid:]
        if swap:
            left, right = right, left
        lr, rr = rect.rectify(left, right)
        _, depth_m = est.compute(lr, rr)
        # NaN (invalid) -> 0, the only invalid marker downstream accepts.
        depth_mm = np.nan_to_num(depth_m * 1000.0, nan=0.0).astype(np.uint16)

        left_enc.stdin.write(cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY).tobytes())
        depth_enc.stdin.write(depth_mm.tobytes())
        valid += float((depth_mm > 0).mean())
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{len(samples)} frames, {n / (time.monotonic() - t0):.1f} fps")

    cap.release()
    for enc in (left_enc, depth_enc):
        enc.stdin.close()
        enc.wait()

    if n != len(samples):
        # The two would pair the wrong timestamp with each frame downstream.
        raise RuntimeError(f"decoded {n} frames but have {len(samples)} timestamps")

    for name in ("dcam_left_timestamps.json", "dcam_depth_timestamps.json"):
        (ep_dir / name).write_text(json.dumps({"samples": samples}))
    K = rect.K
    (ep_dir / "dcam_calib_offline.json").write_text(json.dumps({
        "width": w, "height": h,
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "baseline": rect.baseline_m,
    }, indent=2))
    print(f"  {n} frames in {time.monotonic() - t0:.0f} s, "
          f"mean depth fill {100 * valid / max(n, 1):.0f}%")


def main():
    # argparse, not click like the sibling scripts: this runs in stereokit's
    # environment, which does not have click.
    ap = argparse.ArgumentParser(description="Compute dcam_left/dcam_depth from a stereo_uvc episode.")
    ap.add_argument("-i", "--input_dir", required=True, action="append", type=Path,
                    help="Episode directory (repeatable for batch)")
    ap.add_argument("--max-depth", type=float, default=4.0,
                    help="Depth beyond this (m) is marked invalid (default 4.0)")
    args = ap.parse_args()
    for d in args.input_dir:
        ep_dir = d.expanduser().absolute()
        print(f"Converting {ep_dir.name}...")
        try:
            convert(ep_dir, args.max_depth)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
