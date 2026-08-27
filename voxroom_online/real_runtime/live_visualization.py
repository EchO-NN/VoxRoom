from __future__ import annotations

import json
import math
import os
import select
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class LiveVisualizationConfig:
    enabled: bool = True
    required: bool = True
    update_hz: float = 5.0
    window_name: str = "VoxRoom ZED2i Live"
    panel_width: int = 1800
    panel_height: int = 900
    jpeg_quality: int = 82
    map_min_span_m: float = 6.0
    map_margin_m: float = 0.75

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None) -> "LiveVisualizationConfig":
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            required=bool(data.get("required", True)),
            update_hz=max(0.1, float(data.get("update_hz", 5.0))),
            window_name=str(data.get("window_name", "VoxRoom ZED2i Live")),
            panel_width=max(800, int(data.get("panel_width", 1800))),
            panel_height=max(480, int(data.get("panel_height", 900))),
            jpeg_quality=max(40, min(95, int(data.get("jpeg_quality", 82)))),
            map_min_span_m=max(1.0, float(data.get("map_min_span_m", 6.0))),
            map_margin_m=max(0.0, float(data.get("map_margin_m", 0.75))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "required": bool(self.required),
            "update_hz": float(self.update_hz),
            "window_name": str(self.window_name),
            "panel_size": [int(self.panel_width), int(self.panel_height)],
            "jpeg_quality": int(self.jpeg_quality),
            "map_min_span_m": float(self.map_min_span_m),
            "map_margin_m": float(self.map_margin_m),
        }


class ZedDepthNavVisualizer:
    def __init__(self, config: LiveVisualizationConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.latest_panel_path = self.output_dir / "latest_live_view.jpg"
        self._next_update_at = 0.0
        self._crop_bbox: tuple[int, int, int, int] | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._enabled = bool(config.enabled)
        self._worker_started = False

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    def update(
        self,
        *,
        now: float,
        step: int,
        depth_m: np.ndarray,
        mapper: Any,
        base_pose: Sequence[float],
        tracking_state: str,
        room_count: int,
        mapper_total_ms: float,
    ) -> bool:
        if not self._enabled:
            return False
        period_s = 1.0 / float(self.config.update_hz)
        if self._next_update_at <= 0.0:
            self._next_update_at = float(now)
        if float(now) < self._next_update_at:
            return False
        intervals = math.floor((float(now) - self._next_update_at) / period_s) + 1
        self._next_update_at += intervals * period_s

        projection = getattr(mapper, "last_voxel_navigation_projection", None)
        if projection is None:
            raise RuntimeError("live NavFree visualization requires the nvblox navigation projection")
        nav_free = np.asarray(getattr(projection, "free", None), dtype=bool)
        nav_occupied = np.asarray(getattr(projection, "occupied", None), dtype=bool)
        nav_unknown = np.asarray(getattr(projection, "unknown", None), dtype=bool)
        if nav_free.ndim != 2 or nav_occupied.shape != nav_free.shape or nav_unknown.shape != nav_free.shape:
            raise RuntimeError("nvblox navigation projection has invalid mask shapes")

        panel, self._crop_bbox = render_depth_nav_panel(
            depth_m=depth_m,
            nav_free=nav_free,
            nav_occupied=nav_occupied,
            nav_unknown=nav_unknown,
            map_info=mapper.grid.map_info,
            base_pose=base_pose,
            step=int(step),
            tracking_state=str(tracking_state),
            room_count=int(room_count),
            mapper_total_ms=float(mapper_total_ms),
            panel_size=(int(self.config.panel_width), int(self.config.panel_height)),
            map_min_span_m=float(self.config.map_min_span_m),
            map_margin_m=float(self.config.map_margin_m),
            previous_crop_bbox=self._crop_bbox,
        )
        self._write_panel(panel)
        if self._proc is None:
            self._start_worker()
        if self._proc is None:
            return True
        if self._proc.poll() is not None:
            print("[zed-viz] popup closed; live mapping continues", file=sys.stderr, flush=True)
            self._enabled = False
            self.close()
            return True
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps({"type": "frame_path", "path": str(self.latest_panel_path)}) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            if self.config.required and not self._worker_started:
                raise RuntimeError(f"failed to send the first live visualization frame: {exc}") from exc
            print(f"[zed-viz] popup IPC stopped: {exc}", file=sys.stderr, flush=True)
            self._enabled = False
            self.close()
        return True

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(json.dumps({"type": "close"}) + "\n")
                    proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3.0)
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def _write_panel(self, panel: np.ndarray) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.output_dir / ".latest_live_view.tmp.jpg"
        Image.fromarray(panel).save(
            tmp_path,
            format="JPEG",
            quality=int(self.config.jpeg_quality),
        )
        os.replace(tmp_path, self.latest_panel_path)

    def _start_worker(self) -> None:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self._startup_failure("DISPLAY/WAYLAND_DISPLAY is unavailable")
            return
        cmd = [
            sys.executable,
            "-m",
            "voxroom_online.isaac_runtime.visualization.voxroom_popup_worker",
            "--window-name",
            str(self.config.window_name),
            "--width",
            str(int(self.config.panel_width)),
            "--height",
            str(int(self.config.panel_height)),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("LD_LIBRARY_PATH", None)
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            assert self._proc.stdout is not None
            ready, _, _ = select.select([self._proc.stdout], [], [], 10.0)
            if not ready:
                raise RuntimeError("popup worker readiness timed out")
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"popup worker exited with code {self._proc.poll()}")
            response = json.loads(line)
            if response.get("type") != "ready":
                raise RuntimeError(f"unexpected popup worker response: {response}")
            self._worker_started = True
            print(
                f"[zed-viz] popup ready: {self.config.window_name} "
                f"({self.config.panel_width}x{self.config.panel_height} at {self.config.update_hz:.1f} Hz)",
                flush=True,
            )
        except Exception as exc:
            self.close()
            self._startup_failure(str(exc))

    def _startup_failure(self, reason: str) -> None:
        if self.config.required:
            raise RuntimeError(f"required ZED live visualization could not start: {reason}")
        print(f"[zed-viz] popup disabled: {reason}", file=sys.stderr, flush=True)
        self._enabled = False


def render_depth_nav_panel(
    *,
    depth_m: np.ndarray,
    nav_free: np.ndarray,
    nav_occupied: np.ndarray,
    nav_unknown: np.ndarray,
    map_info: Any,
    base_pose: Sequence[float],
    step: int,
    tracking_state: str,
    room_count: int,
    mapper_total_ms: float,
    panel_size: tuple[int, int] = (1800, 900),
    map_min_span_m: float = 6.0,
    map_margin_m: float = 0.75,
    previous_crop_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    free = np.asarray(nav_free, dtype=bool)
    occupied = np.asarray(nav_occupied, dtype=bool)
    unknown = np.asarray(nav_unknown, dtype=bool)
    if free.ndim != 2 or occupied.shape != free.shape or unknown.shape != free.shape:
        raise ValueError("NavFree, occupied, and unknown masks must have the same 2D shape")
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array")

    width, height = int(panel_size[0]), int(panel_size[1])
    if width < 800 or height < 480:
        raise ValueError("live visualization panel must be at least 800x480")
    header_h = 68
    title_h = 42
    footer_h = 46
    outer = 22
    gap = 24
    content_top = header_h + title_h
    content_bottom = height - footer_h
    half_w = (width - 2 * outer - gap) // 2
    left_box = (outer, content_top, outer + half_w, content_bottom)
    right_box = (outer + half_w + gap, content_top, width - outer, content_bottom)

    panel = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    font_regular = _font(22, bold=False)
    font_title = _font(28, bold=True)
    font_header = _font(25, bold=True)
    draw.text(
        (outer, 18),
        "ZED2i LIVE  |  step %d  |  tracking %s  |  map %.1f ms  |  rooms %d"
        % (int(step), str(tracking_state), float(mapper_total_ms), int(room_count)),
        fill=(28, 32, 36),
        font=font_header,
    )
    draw.text((left_box[0], header_h), "DEPTH", fill=(20, 24, 28), font=font_title)
    draw.text((right_box[0], header_h), "NVBLOX NAVFREE", fill=(20, 24, 28), font=font_title)

    depth_rgb, depth_summary = _render_depth(depth)
    _paste_fitted(panel, Image.fromarray(depth_rgb), left_box, Image.Resampling.BILINEAR)

    robot_row, robot_col = _world_to_grid(float(base_pose[0]), float(base_pose[1]), map_info)
    crop_bbox = _navigation_crop_bbox(
        free=free,
        occupied=occupied,
        unknown=unknown,
        robot_cell=(robot_row, robot_col),
        resolution_m=float(map_info.resolution_m),
        min_span_m=float(map_min_span_m),
        margin_m=float(map_margin_m),
        previous=previous_crop_bbox,
    )
    map_rgb = _render_navigation_crop(
        free=free,
        occupied=occupied,
        unknown=unknown,
        crop_bbox=crop_bbox,
        robot_cell=(robot_row, robot_col),
        robot_yaw=float(base_pose[3]),
        resolution_m=float(map_info.resolution_m),
    )
    _paste_fitted(panel, Image.fromarray(map_rgb), right_box, Image.Resampling.NEAREST)

    draw.rectangle(left_box, outline=(115, 120, 124), width=2)
    draw.rectangle(right_box, outline=(115, 120, 124), width=2)
    known = free | occupied | ~unknown
    footer_y = height - footer_h + 9
    draw.text(
        (outer, footer_y),
        "%s   |   valid %.1f%%" % (depth_summary, 100.0 * float(np.count_nonzero(np.isfinite(depth) & (depth > 0.0))) / float(depth.size)),
        fill=(45, 50, 55),
        font=font_regular,
    )
    nav_text = "FREE %d   OCCUPIED %d   UNKNOWN %d   OBSERVED %d" % (
        int(np.count_nonzero(free)),
        int(np.count_nonzero(occupied)),
        int(np.count_nonzero(unknown)),
        int(np.count_nonzero(known)),
    )
    draw.text((right_box[0], footer_y), nav_text, fill=(45, 50, 55), font=font_regular)
    return np.asarray(panel, dtype=np.uint8), crop_bbox


def _render_depth(depth: np.ndarray) -> tuple[np.ndarray, str]:
    valid = np.isfinite(depth) & (depth > 0.0)
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return rgb, "no valid depth"
    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    hi = max(float(hi), float(lo) + 1.0e-3)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip((depth[valid] - float(lo)) / (hi - float(lo)), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    value = np.asarray(np.rint((1.0 - normalized) * 235.0 + 20.0), dtype=np.uint8)
    rgb[valid] = np.stack([value[valid], value[valid], value[valid]], axis=-1)
    return rgb, "range %.2f-%.2f m" % (float(lo), float(hi))


def _render_navigation_crop(
    *,
    free: np.ndarray,
    occupied: np.ndarray,
    unknown: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    robot_cell: tuple[int, int],
    robot_yaw: float,
    resolution_m: float,
) -> np.ndarray:
    r0, r1, c0, c1 = crop_bbox
    rgb = np.full((r1 - r0, c1 - c0, 3), (158, 162, 166), dtype=np.uint8)
    local_unknown = unknown[r0:r1, c0:c1]
    local_free = free[r0:r1, c0:c1]
    local_occupied = occupied[r0:r1, c0:c1]
    rgb[~local_unknown] = (205, 208, 210)
    rgb[local_free & ~local_occupied] = (252, 252, 252)
    rgb[local_occupied] = (18, 20, 22)

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    rr = int(robot_cell[0]) - r0
    cc = int(robot_cell[1]) - c0
    if 0 <= rr < rgb.shape[0] and 0 <= cc < rgb.shape[1]:
        radius = max(2, int(round(0.14 / max(float(resolution_m), 1.0e-6))))
        draw.ellipse((cc - radius, rr - radius, cc + radius, rr + radius), fill=(220, 36, 42))
        length = max(radius + 2, int(round(0.45 / max(float(resolution_m), 1.0e-6))))
        tip_c = cc + int(round(length * math.cos(float(robot_yaw))))
        tip_r = rr - int(round(length * math.sin(float(robot_yaw))))
        draw.line((cc, rr, tip_c, tip_r), fill=(220, 36, 42), width=max(2, radius // 2))
    return np.asarray(image, dtype=np.uint8)


def _navigation_crop_bbox(
    *,
    free: np.ndarray,
    occupied: np.ndarray,
    unknown: np.ndarray,
    robot_cell: tuple[int, int],
    resolution_m: float,
    min_span_m: float,
    margin_m: float,
    previous: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    height, width = free.shape
    known = free | occupied | ~unknown
    rows, cols = np.nonzero(known)
    robot_row, robot_col = robot_cell
    if rows.size:
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
    else:
        r0, r1 = robot_row, robot_row + 1
        c0, c1 = robot_col, robot_col + 1
    if 0 <= robot_row < height and 0 <= robot_col < width:
        r0, r1 = min(r0, robot_row), max(r1, robot_row + 1)
        c0, c1 = min(c0, robot_col), max(c1, robot_col + 1)
    margin_cells = int(math.ceil(float(margin_m) / float(resolution_m)))
    r0, r1 = r0 - margin_cells, r1 + margin_cells
    c0, c1 = c0 - margin_cells, c1 + margin_cells
    if previous is not None:
        pr0, pr1, pc0, pc1 = previous
        r0, r1 = min(r0, pr0), max(r1, pr1)
        c0, c1 = min(c0, pc0), max(c1, pc1)
    min_cells = max(1, int(math.ceil(float(min_span_m) / float(resolution_m))))
    span = max(min_cells, r1 - r0, c1 - c0)
    center_r = 0.5 * (r0 + r1)
    center_c = 0.5 * (c0 + c1)
    return _clamped_square_bbox(center_r, center_c, span, height, width)


def _clamped_square_bbox(
    center_r: float,
    center_c: float,
    span: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    span = max(1, min(int(span), int(height), int(width)))
    r0 = int(round(float(center_r) - span * 0.5))
    c0 = int(round(float(center_c) - span * 0.5))
    r0 = max(0, min(int(height) - span, r0))
    c0 = max(0, min(int(width) - span, c0))
    return r0, r0 + span, c0, c0 + span


def _world_to_grid(x: float, y: float, map_info: Any) -> tuple[int, int]:
    col = int(math.floor((float(x) - float(map_info.min_x)) / float(map_info.resolution_m)))
    row = int(math.floor((float(map_info.max_y) - float(y)) / float(map_info.resolution_m)))
    return row, col


def _paste_fitted(panel: Image.Image, source: Image.Image, box: tuple[int, int, int, int], resampling: int) -> None:
    x0, y0, x1, y1 = box
    max_w = max(1, x1 - x0)
    max_h = max(1, y1 - y0)
    scale = min(max_w / float(source.width), max_h / float(source.height))
    target_w = max(1, int(round(source.width * scale)))
    target_h = max(1, int(round(source.height * scale)))
    resized = source.resize((target_w, target_h), resample=resampling)
    paste_x = x0 + (max_w - target_w) // 2
    paste_y = y0 + (max_h - target_h) // 2
    panel.paste(resized, (paste_x, paste_y))


def _font(size: int, *, bold: bool) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, int(size))
    except OSError:
        return ImageFont.load_default()
