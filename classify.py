"""Webcam CLI for battery vs wire waste detection."""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from classifier import DEFAULT_THRESHOLD
from detector import (
    DEFAULT_DET_CONF,
    DEFAULT_MIN_SIZE,
    Detection,
    WasteDetector,
    center_crop_box,
)

CLASSIFY_HZ = 2.0
COLORS = {
    "battery": (0, 140, 255),
    "wire": (60, 200, 80),
    "invalid": (80, 80, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect batteries and wires in the webcam with boxes.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum CLIP confidence on a crop (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=DEFAULT_MIN_SIZE,
        help=(
            "How much of the frame the object must cover, as a distance stand-in; "
            f"raise to require it closer (default: {DEFAULT_MIN_SIZE})"
        ),
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=DEFAULT_DET_CONF,
        help=f"Optional YOLO-World box confidence (default: {DEFAULT_DET_CONF})",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not show the OpenCV preview window (quit with Ctrl+C)",
    )
    return parser.parse_args()


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. "
            "Try --camera 1 or close other apps using the webcam."
        )
    return cap


def draw_detections(frame, detections: list[Detection], reason: str = ""):
    overlay = frame.copy()
    if not detections:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = center_crop_box(width, height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(
            overlay,
            reason or "none",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    for det in detections:
        x1, y1, x2, y2 = det.box
        color = COLORS.get(det.label, (255, 255, 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        caption = f"{det.label} {det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            caption,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )
        top = max(0, y1 - text_h - 10)
        cv2.rectangle(overlay, (x1, top), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(
            overlay,
            caption,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return overlay


def print_detections(detections: list[Detection], reason: str = "") -> None:
    if not detections:
        print(reason or "none", flush=True)
        return
    for det in detections:
        x1, y1, x2, y2 = det.box
        print(f"{det.label}  {det.confidence:.2f}  [{x1},{y1},{x2},{y2}]", flush=True)


def main() -> int:
    args = parse_args()
    print("Loading CLIP (first run downloads weights)...")
    detector = WasteDetector()
    quit_hint = "Press q or Esc to quit." if not args.no_preview else "Press Ctrl+C to quit."
    print(f"Model ready. Hold a battery or wire in the center of the camera. {quit_hint}")

    try:
        cap = open_camera(args.camera)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    last_print = 0.0
    classify_interval = 1.0 / CLASSIFY_HZ
    last_detections: list[Detection] = []
    last_reason = ""
    window_name = "Waste detector"

    if not args.no_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.", file=sys.stderr)
                return 1

            now = time.monotonic()
            if now - last_print >= classify_interval:
                last_detections, last_reason = detector.detect_with_reason(
                    frame,
                    threshold=args.threshold,
                    det_conf=args.det_conf,
                    min_size=args.min_size,
                )
                print_detections(last_detections, last_reason)
                last_print = now

            if args.no_preview:
                time.sleep(0.01)
                continue

            overlay = draw_detections(frame, last_detections, last_reason)
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
