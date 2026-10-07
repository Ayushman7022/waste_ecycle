"""Local web app: live webcam, battery/wire labels, save a shot when a new object appears."""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from classify import CLASSIFY_HZ, draw_detections, open_camera
from detector import DEFAULT_DET_CONF, DEFAULT_MIN_SIZE, Detection, WasteDetector

ROOT = Path(__file__).resolve().parent
CAPTURE_ROOT = ROOT / "captures"
LABELS = ("battery", "wire")
STABLE_SECONDS = 0.5
SAVE_GAP_SECONDS = 2.0
RECENT_LIMIT = 8
CAMERA_INDEX = int(os.environ.get("CAMERA", "0"))


class SaveGate:
    """Save once per new stable battery/wire, not on every frame of the same object."""

    def __init__(self) -> None:
        self.candidate = ""
        self.candidate_since = 0.0
        self.stable = ""
        self.pending = False
        self.last_save = float("-inf")

    def update(self, label: str, now: float) -> str | None:
        if label != self.candidate:
            self.candidate = label
            self.candidate_since = now
            return None
        if now - self.candidate_since < STABLE_SECONDS:
            return None
        if label != self.stable:
            self.stable = label
            self.pending = label in LABELS
        if not self.pending or now - self.last_save < SAVE_GAP_SECONDS:
            return None
        self.pending = False
        self.last_save = now
        return label


def top_detection(detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    return max(detections, key=lambda item: item.confidence)


def frame_label(detections: list[Detection], reason: str) -> tuple[str, float]:
    found = top_detection(detections)
    if found is None:
        return reason or "none", 0.0
    return found.label, found.confidence


class CameraSession:
    def __init__(self, detector: WasteDetector, camera_index: int = CAMERA_INDEX) -> None:
        self.detector = detector
        self.camera_index = camera_index
        self.lock = threading.Lock()
        self.stopped = False
        self.thread: threading.Thread | None = None
        self.jpeg: bytes | None = None
        self.label = "none"
        self.confidence = 0.0
        self.reason = ""
        self.error = ""
        self.last_saved = ""
        self.gate = SaveGate()
        for label in LABELS:
            (CAPTURE_ROOT / label).mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="webcam", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def status(self) -> dict:
        with self.lock:
            label = self.label
            confidence = self.confidence
            reason = self.reason
            error = self.error
            last_saved = self.last_saved
            camera_ok = self.jpeg is not None and not error
        counts = {}
        recent = {}
        for name in LABELS:
            files = _recent_files(name)
            counts[name] = _count_files(name)
            recent[name] = files
        return {
            "label": label,
            "confidence": round(confidence, 2),
            "reason": reason,
            "last_saved": last_saved,
            "counts": counts,
            "recent": recent,
            "camera_ok": camera_ok,
            "error": error,
        }

    def _run(self) -> None:
        try:
            cap = open_camera(self.camera_index)
        except RuntimeError as exc:
            with self.lock:
                self.error = str(exc)
            return

        detections: list[Detection] = []
        reason = ""
        last_classify = 0.0
        interval = 1.0 / CLASSIFY_HZ
        try:
            while not self.stopped:
                ok, frame = cap.read()
                if not ok:
                    with self.lock:
                        self.error = "Failed to read a frame from the camera."
                    break

                now = time.monotonic()
                if now - last_classify >= interval:
                    detections, reason = self.detector.detect_with_reason(
                        frame,
                        min_size=DEFAULT_MIN_SIZE,
                        det_conf=DEFAULT_DET_CONF,
                    )
                    self._maybe_save(frame, detections, reason, now)
                    last_classify = now

                overlay = draw_detections(frame, detections, reason)
                encoded, buffer = cv2.imencode(
                    ".jpg",
                    overlay,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                )
                if not encoded:
                    continue
                label, confidence = frame_label(detections, reason)
                with self.lock:
                    self.jpeg = buffer.tobytes()
                    self.label = label
                    self.confidence = confidence
                    self.reason = reason
                    self.error = ""
        finally:
            cap.release()

    def _maybe_save(
        self,
        frame,
        detections: list[Detection],
        reason: str,
        now: float,
    ) -> None:
        label, confidence = frame_label(detections, reason)
        if self.gate.update(label, now) is None:
            return
        overlay = draw_detections(frame, detections, reason)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stamp}_{label}_{confidence:.2f}.jpg"
        path = CAPTURE_ROOT / label / filename
        if not cv2.imwrite(str(path), overlay):
            return
        relative = path.relative_to(ROOT).as_posix()
        with self.lock:
            self.last_saved = relative
        print(f"saved {relative}", flush=True)

    def latest_jpeg(self) -> bytes | None:
        with self.lock:
            return self.jpeg


def _recent_files(label: str) -> list[dict]:
    folder = CAPTURE_ROOT / label
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*.jpg"), key=lambda item: item.stat().st_mtime, reverse=True)
    recent = []
    for path in files[:RECENT_LIMIT]:
        recent.append(
            {
                "name": path.name,
                "url": f"/captures/{label}/{path.name}",
            }
        )
    return recent


def _count_files(label: str) -> int:
    folder = CAPTURE_ROOT / label
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.glob("*.jpg"))


def _safe_capture(label: str, name: str) -> Path:
    if label not in LABELS or "/" in name or "\\" in name or name != Path(name).name:
        raise HTTPException(status_code=404)
    path = (CAPTURE_ROOT / label / name).resolve()
    if path.parent != (CAPTURE_ROOT / label).resolve() or not path.is_file():
        raise HTTPException(status_code=404)
    return path


session: CameraSession | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global session
    print("Loading CLIP (first run downloads weights)...")
    detector = WasteDetector()
    print("Model ready. Open http://127.0.0.1:8000")
    session = CameraSession(detector)
    session.start()
    yield
    session.stop()
    session = None


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(page)


@app.get("/video")
def video() -> StreamingResponse:
    return StreamingResponse(_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def _frames():
    while True:
        current = session.latest_jpeg() if session is not None else None
        if current is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + current + b"\r\n"
        )
        time.sleep(0.05)


@app.get("/status")
def status() -> dict:
    if session is None:
        return {
            "label": "none",
            "confidence": 0.0,
            "reason": "",
            "last_saved": "",
            "counts": {"battery": 0, "wire": 0},
            "recent": {"battery": [], "wire": []},
            "camera_ok": False,
            "error": "Camera is not running.",
        }
    return session.status()


@app.get("/captures/{label}/{name}")
def capture_file(label: str, name: str) -> Response:
    path = _safe_capture(label, name)
    return Response(content=path.read_bytes(), media_type="image/jpeg")
