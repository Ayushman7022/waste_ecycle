"""OpenCV + CLIP battery/wire detector. YOLO-World is optional extra proposals."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from classifier import DEFAULT_THRESHOLD, WasteClassifier

DETECT_PROMPTS = [
    "battery",
    "AA battery",
    "electrical wire",
    "cable",
]

DEFAULT_DET_CONF = 0.05
DEFAULT_MIN_SIDE = 16
# Near-field window. A small window keeps a distant object from being judged
# together with a wide view of the room behind it.
CENTER_CROP_RATIO = 0.45
# A webcam cannot measure distance, so apparent size stands in for it. With a
# typical ~60 degree lens, a 10 cm object one metre away covers about 12% of the
# frame and a 5 cm battery about 6%, so 0.10 sits near a one metre cut-off.
DEFAULT_MIN_SIZE = 0.10
CROP_PAD = 0.06
CONTAINMENT_THRESH = 0.7
MIN_TIGHTER_AREA_RATIO = 0.04
WORLD_WEIGHTS = "yolov8s-worldv2.pt"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class WasteDetector:
    """Proposes boxes with OpenCV (and optional YOLO), then CLIP-labels each crop."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = WasteClassifier(device=self.device)
        self.model = None
        try:
            from ultralytics import YOLOWorld

            self.model = YOLOWorld(WORLD_WEIGHTS)
            self.model.set_classes(DETECT_PROMPTS)
        except Exception:
            try:
                from ultralytics import YOLO

                self.model = YOLO(WORLD_WEIGHTS)
                self.model.set_classes(DETECT_PROMPTS)
            except Exception:
                self.model = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
        det_conf: float = DEFAULT_DET_CONF,
        min_size: float = DEFAULT_MIN_SIZE,
    ) -> list[Detection]:
        return self.detect_with_reason(frame_bgr, threshold, det_conf, min_size)[0]

    def detect_with_reason(
        self,
        frame_bgr: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
        det_conf: float = DEFAULT_DET_CONF,
        min_size: float = DEFAULT_MIN_SIZE,
    ) -> tuple[list[Detection], str]:
        """Reason is 'too far' when the distance gate stopped the frame early."""
        height, width = frame_bgr.shape[:2]
        contour = _contour_box(frame_bgr)
        line = _line_box(frame_bgr)

        # Distance gate. The proposal boxes are fixed windows, so they say nothing
        # about range; the foreground blob is what shrinks as the object moves away.
        if min_size > 0 and not _is_near_enough(contour, line, width, height, min_size):
            return [], "too far"

        boxes = _proposal_boxes(frame_bgr, contour, line)
        boxes.extend(self._yolo_boxes(frame_bgr, det_conf))
        boxes = _unique_boxes(boxes)

        detections: list[Detection] = []
        for box in boxes:
            crop = _padded_crop(frame_bgr, box, CROP_PAD)
            result = self.classifier.classify(crop, threshold=threshold)
            if result.label not in ("battery", "wire", "invalid"):
                continue
            detections.append(
                Detection(
                    label=result.label,
                    confidence=result.confidence,
                    box=box,
                )
            )
        # A real battery or wire in any crop wins over an overlapping "other" object.
        valid = [item for item in detections if item.label in ("battery", "wire")]
        if valid:
            return _nms(valid), ""
        invalid = [item for item in detections if item.label == "invalid"]
        if invalid:
            return _nms(invalid), ""
        return [], ""

    def _yolo_boxes(self, frame_bgr: np.ndarray, det_conf: float) -> list[tuple[int, int, int, int]]:
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=det_conf,
                verbose=False,
                device=self.device,
            )
        except Exception:
            return []
        if not results or results[0].boxes is None:
            return []
        height, width = frame_bgr.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            clipped = _clip_box((x1, y1, x2, y2), width, height)
            if clipped is not None:
                boxes.append(clipped)
        return boxes


def center_crop_box(width: int, height: int, ratio: float = CENTER_CROP_RATIO) -> tuple[int, int, int, int]:
    box_w = int(width * ratio)
    box_h = int(height * ratio)
    x1 = (width - box_w) // 2
    y1 = (height - box_h) // 2
    return (x1, y1, x1 + box_w, y1 + box_h)


def _is_near_enough(
    contour: tuple[int, int, int, int] | None,
    line: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    min_size: float,
) -> bool:
    """A wire is long and thin, so one big dimension is enough to count as near."""
    for box in (contour, line):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1) >= min_size * width or (y2 - y1) >= min_size * height:
            return True
    return False


def _proposal_boxes(
    frame_bgr: np.ndarray,
    contour: tuple[int, int, int, int] | None,
    line: tuple[int, int, int, int] | None,
) -> list[tuple[int, int, int, int]]:
    height, width = frame_bgr.shape[:2]
    boxes = [center_crop_box(width, height)]
    for candidate in (contour, line):
        if candidate is not None:
            boxes.append(candidate)
    return boxes


def _contour_box(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    height, width = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(height * width)
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        if area < 0.01 * frame_area or area > 0.95 * frame_area:
            continue
        if area > best_area:
            best_area = area
            best = _clip_box((x, y, x + box_w, y + box_h), width, height)
    return best


def _line_box(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    height, width = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    min_length = int(min(height, width) * 0.25)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=60,
        minLineLength=min_length,
        maxLineGap=20,
    )
    if lines is None:
        return None

    xs: list[int] = []
    ys: list[int] = []
    for line in lines:
        x1, y1, x2, y2 = (int(v) for v in line[0])
        if float(np.hypot(x2 - x1, y2 - y1)) < min_length:
            continue
        xs.extend((x1, x2))
        ys.extend((y1, y2))
    if not xs:
        return None

    pad = 12
    return _clip_box(
        (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad),
        width,
        height,
    )


def _clip_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if (x2 - x1) < DEFAULT_MIN_SIDE or (y2 - y1) < DEFAULT_MIN_SIDE:
        return None
    return (x1, y1, x2, y2)


def _padded_crop(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    pad_ratio: float,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return frame_bgr[y1:y2, x1:x2]


def _unique_boxes(
    boxes: list[tuple[int, int, int, int]],
    iou_thresh: float = 0.8,
) -> list[tuple[int, int, int, int]]:
    unique: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if box is None:
            continue
        if all(_iou(box, kept) < iou_thresh for kept in unique):
            unique.append(box)
    return unique


def _nms(detections: list[Detection], iou_thresh: float = 0.5) -> list[Detection]:
    if len(detections) <= 1:
        return detections
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        overlap_idx = next(
            (
                i
                for i, item in enumerate(kept)
                if _iou(candidate.box, item.box) >= iou_thresh
                or _containment(candidate.box, item.box) >= CONTAINMENT_THRESH
            ),
            None,
        )
        if overlap_idx is None:
            kept.append(candidate)
            continue
        # The center crop and a tight contour box describe the same object, so
        # show the tighter one instead of a second box over the same pixels.
        existing = kept[overlap_idx]
        if (
            candidate.label == existing.label
            and _area(candidate.box) < _area(existing.box)
            and _area(candidate.box) >= MIN_TIGHTER_AREA_RATIO * _area(existing.box)
        ):
            kept[overlap_idx] = candidate
    return kept


def _area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _containment(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Fraction of the smaller box that sits inside the other box."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    smaller = min(_area(a), _area(b))
    if inter == 0 or smaller == 0:
        return 0.0
    return inter / float(smaller)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter)
