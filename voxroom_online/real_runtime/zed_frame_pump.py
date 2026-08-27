from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from voxroom_online.real_runtime.zed2i_source import ZedFrame


@dataclass(frozen=True)
class ZedFramePumpStats:
    capture_frames: int
    depth_frames_published: int
    depth_frames_dropped: int
    failed_grabs: int
    elapsed_s: float
    capture_fps_average: float
    capture_fps_recent: float
    last_frame_age_s: float | None
    finished: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ZedFramePump:
    """Keep ZED VIO at camera rate while publishing only fresh mapping frames."""

    def __init__(
        self,
        source: Any,
        *,
        depth_hz: float,
        minimum_capture_fps: float,
        fps_window_s: float,
        low_fps_timeout_s: float,
    ) -> None:
        if float(depth_hz) <= 0.0:
            raise ValueError("depth_hz must be positive")
        if float(minimum_capture_fps) < 0.0:
            raise ValueError("minimum_capture_fps must be non-negative")
        if float(fps_window_s) <= 0.0:
            raise ValueError("fps_window_s must be positive")
        if float(low_fps_timeout_s) <= 0.0:
            raise ValueError("low_fps_timeout_s must be positive")

        self.source = source
        self.depth_hz = float(depth_hz)
        self.minimum_capture_fps = float(minimum_capture_fps)
        self.fps_window_s = float(fps_window_s)
        self.low_fps_timeout_s = float(low_fps_timeout_s)
        self._frames: queue.Queue[ZedFrame] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._done_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._capture_frames = 0
        self._depth_frames_published = 0
        self._depth_frames_dropped = 0
        self._failed_grabs = 0
        self._started_at = 0.0
        self._first_frame_at = 0.0
        self._last_frame_at = 0.0
        self._recent_frame_times: deque[float] = deque()

    @property
    def finished(self) -> bool:
        return self._done_event.is_set()

    def start(self, *, timeout_s: float = 30.0) -> None:
        if self._thread is not None:
            raise RuntimeError("ZED frame pump has already been started")
        self._thread = threading.Thread(target=self._run, name="zed2i-acquisition", daemon=False)
        self._thread.start()
        if not self._ready_event.wait(timeout=max(0.1, float(timeout_s))):
            self.stop()
            self.join(timeout_s=5.0)
            raise RuntimeError("timed out while opening the ZED camera in the acquisition thread")
        self.raise_if_failed()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, *, timeout_s: float | None = None) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=None if timeout_s is None else max(0.0, float(timeout_s)))
        if thread.is_alive():
            raise RuntimeError("ZED acquisition thread did not stop")

    def get(self, *, timeout_s: float = 0.1) -> ZedFrame | None:
        self.raise_if_failed()
        try:
            frame = self._frames.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            self.raise_if_failed()
            return None
        self.raise_if_failed()
        return frame

    def raise_if_failed(self) -> None:
        error = self._error
        if error is not None:
            raise RuntimeError(f"ZED acquisition failed: {error}") from error

    def stats(self) -> ZedFramePumpStats:
        now = time.monotonic()
        with self._lock:
            elapsed_s = max(0.0, now - self._started_at) if self._started_at > 0.0 else 0.0
            average_fps = self._rate(
                self._capture_frames,
                self._first_frame_at,
                self._last_frame_at,
            )
            recent_fps = self._timestamp_rate(self._recent_frame_times)
            last_frame_age_s = (
                max(0.0, now - self._last_frame_at) if self._last_frame_at > 0.0 else None
            )
            return ZedFramePumpStats(
                capture_frames=self._capture_frames,
                depth_frames_published=self._depth_frames_published,
                depth_frames_dropped=self._depth_frames_dropped,
                failed_grabs=self._failed_grabs,
                elapsed_s=elapsed_s,
                capture_fps_average=average_fps,
                capture_fps_recent=recent_fps,
                last_frame_age_s=last_frame_age_s,
                finished=self._done_event.is_set(),
            )

    def _run(self) -> None:
        opened = False
        try:
            self.source.open()
            opened = True
            actual_fps = float(dict(getattr(self.source, "camera_info", {}) or {}).get("fps", 0.0) or 0.0)
            if self.minimum_capture_fps > 0.0 and actual_fps > 0.0 and actual_fps < self.minimum_capture_fps:
                raise RuntimeError(
                    "camera negotiated %.1f FPS, below required %.1f FPS"
                    % (actual_fps, self.minimum_capture_fps)
                )
            with self._lock:
                self._started_at = time.monotonic()
            self._ready_event.set()

            depth_period_s = 1.0 / self.depth_hz
            next_depth_at = time.monotonic()
            low_fps_started_at: float | None = None
            while not self._stop_event.is_set():
                before_grab = time.monotonic()
                retrieve_depth = before_grab >= next_depth_at
                if retrieve_depth:
                    intervals = int((before_grab - next_depth_at) // depth_period_s) + 1
                    next_depth_at += intervals * depth_period_s

                frame = self.source.grab(retrieve_depth=retrieve_depth)
                captured_at = time.monotonic()
                if frame is None:
                    with self._lock:
                        self._failed_grabs += 1
                    if bool(getattr(self.source, "finished", False)):
                        break
                    continue

                with self._lock:
                    self._capture_frames += 1
                    if self._first_frame_at <= 0.0:
                        self._first_frame_at = captured_at
                    self._last_frame_at = captured_at
                    self._recent_frame_times.append(captured_at)
                    cutoff = captured_at - self.fps_window_s
                    while self._recent_frame_times and self._recent_frame_times[0] < cutoff:
                        self._recent_frame_times.popleft()
                    recent_fps = self._timestamp_rate(self._recent_frame_times)
                    measurement_age_s = captured_at - self._first_frame_at

                if retrieve_depth:
                    self._publish_latest(frame)

                if (
                    self.minimum_capture_fps > 0.0
                    and measurement_age_s >= self.fps_window_s
                    and recent_fps < self.minimum_capture_fps
                ):
                    if low_fps_started_at is None:
                        low_fps_started_at = captured_at
                    elif captured_at - low_fps_started_at >= self.low_fps_timeout_s:
                        raise RuntimeError(
                            "effective ZED tracking rate stayed at %.1f FPS, below required %.1f FPS for %.1fs"
                            % (recent_fps, self.minimum_capture_fps, self.low_fps_timeout_s)
                        )
                else:
                    low_fps_started_at = None
        except BaseException as exc:
            self._error = exc
            self._ready_event.set()
        finally:
            if opened:
                try:
                    self.source.close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
            self._ready_event.set()
            self._done_event.set()

    def _publish_latest(self, frame: ZedFrame) -> None:
        dropped = 0
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
                dropped = 1
            except queue.Empty:
                pass
            self._frames.put_nowait(frame)
        with self._lock:
            self._depth_frames_published += 1
            self._depth_frames_dropped += dropped

    @staticmethod
    def _rate(count: int, first_at: float, last_at: float) -> float:
        duration_s = float(last_at) - float(first_at)
        if int(count) < 2 or duration_s <= 0.0:
            return 0.0
        return float(int(count) - 1) / duration_s

    @staticmethod
    def _timestamp_rate(timestamps: deque[float]) -> float:
        if len(timestamps) < 2:
            return 0.0
        duration_s = float(timestamps[-1]) - float(timestamps[0])
        if duration_s <= 0.0:
            return 0.0
        return float(len(timestamps) - 1) / duration_s
