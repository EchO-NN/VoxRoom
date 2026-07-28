import hashlib
import json
import os
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from run_context_contract import (
    STRICT_CAPTURE_MARKER_HEIGHT,
    STRICT_CAPTURE_MARKER_WIDTH,
    STRICT_CAPTURE_MARKER_X,
    STRICT_CAPTURE_MARKER_Y,
    strict_capture_marker_bits,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, payload):
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class TopologyEventRecorder:
    def __init__(self, path, run_id, process_id, latest_event_limit=8):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError("Topology event file already exists: {}".format(self.path))
        self.run_id = str(run_id)
        self.process_id = int(process_id)
        self.sequence = 0
        self.counts = Counter()
        self.latest = deque(maxlen=int(latest_event_limit))

    def record(self, event_type, step, **payload):
        self.sequence += 1
        event = {
            "run_id": self.run_id,
            "process_id": self.process_id,
            "sequence": self.sequence,
            "step": int(step),
            "event_type": str(event_type),
            "timestamp_unix": time.time(),
            "payload": _json_value(payload),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.counts[event["event_type"]] += 1
        self.latest.append(event)
        return event

    def latest_events(self):
        return list(self.latest)

    def summary(self):
        return {
            "event_count": self.sequence,
            "event_counts": dict(sorted(self.counts.items())),
            "event_file": self.path.name,
        }


class RuntimeDashboard:
    FIXED_RENDER_SIZE_INCHES = (16.0, 9.0)
    FRAME_DPI = 100
    FINAL_DPI = 120
    STATUS_COLORS = {
        "exploring": "#16a085",
        "explored": "#7f8c8d",
        "unexplored": "#f39c12",
    }

    def __init__(
        self,
        figure,
        run_dir,
        capture_identity,
        frame_every_steps=10,
        refresh_seconds=0.01,
    ):
        self.figure = figure
        self.run_dir = Path(run_dir)
        self.capture_identity = str(capture_identity)
        if not self.capture_identity:
            raise ValueError("Runtime dashboard requires a capture identity")
        self.frame_dir = self.run_dir / "visualization_frames"
        self.frame_every_steps = int(frame_every_steps)
        self.refresh_seconds = float(refresh_seconds)
        self.last_saved_step = -1
        self.frame_paths = []
        self.last_render_step = 0
        self.capture_marker_artists = []

        self.figure.clf()
        grid = self.figure.add_gridspec(
            2,
            2,
            left=0.035,
            right=0.985,
            bottom=0.055,
            top=0.94,
            wspace=0.12,
            hspace=0.2,
        )
        self.rgb_axis = self.figure.add_subplot(grid[0, 0])
        self.map_axis = self.figure.add_subplot(grid[0, 1])
        self.topology_axis = self.figure.add_subplot(grid[1, 0])
        status_grid = grid[1, 1].subgridspec(
            2,
            1,
            height_ratios=[1.0, 1.4],
            hspace=0.32,
        )
        self.metrics_axis = self.figure.add_subplot(status_grid[0, 0])
        self.events_axis = self.figure.add_subplot(status_grid[1, 0])

    def _draw_capture_marker(self, step):
        for artist in self.capture_marker_artists:
            artist.remove()
        self.capture_marker_artists = []
        bits = strict_capture_marker_bits(self.capture_identity, step)
        bit_width = STRICT_CAPTURE_MARKER_WIDTH / len(bits)
        for index, bit in enumerate(bits):
            artist = Rectangle(
                (
                    STRICT_CAPTURE_MARKER_X + index * bit_width,
                    STRICT_CAPTURE_MARKER_Y,
                ),
                bit_width,
                STRICT_CAPTURE_MARKER_HEIGHT,
                transform=self.figure.transFigure,
                facecolor="#111111" if bit else "#f4f4f4",
                edgecolor="none",
                linewidth=0,
                clip_on=False,
                zorder=1000,
            )
            self.figure.add_artist(artist)
            self.capture_marker_artists.append(artist)

    def _save_fixed_render(self, path, dpi):
        original_size = self.figure.get_size_inches().copy()
        try:
            self.figure.set_size_inches(
                *self.FIXED_RENDER_SIZE_INCHES,
                forward=False,
            )
            self.figure.canvas.draw()
            self.figure.savefig(path, dpi=dpi, facecolor="white")
        finally:
            self.figure.set_size_inches(*original_size, forward=False)
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()

    @staticmethod
    def _map_image(occupied, explored):
        occupied = np.asarray(occupied) > 0.5
        explored = np.asarray(explored) > 0.5
        image = np.empty(occupied.shape + (3,), dtype=np.float32)
        image[:] = (0.84, 0.86, 0.88)
        image[explored] = (0.98, 0.98, 0.97)
        image[occupied] = (0.12, 0.13, 0.15)
        return image

    @staticmethod
    def _node_positions(nodes):
        count = len(nodes)
        if count == 0:
            return {}
        if count == 1:
            return {nodes[0]["id"]: (0.5, 0.5)}
        positions = {}
        for index, node in enumerate(nodes):
            angle = -np.pi / 2.0 + 2.0 * np.pi * index / count
            positions[node["id"]] = (
                0.5 + 0.36 * np.cos(angle),
                0.5 + 0.36 * np.sin(angle),
            )
        return positions

    def _draw_topology(self, snapshot):
        axis = self.topology_axis
        axis.clear()
        axis.set_title("Room topology")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal")
        axis.axis("off")

        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])
        positions = self._node_positions(nodes)
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            if source not in positions or target not in positions:
                continue
            source_xy = positions[source]
            target_xy = positions[target]
            curve = 0.14 if source <= target else -0.14
            axis.annotate(
                "",
                xy=target_xy,
                xytext=source_xy,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#52616b",
                    "linewidth": 1.6,
                    "shrinkA": 18,
                    "shrinkB": 18,
                    "connectionstyle": "arc3,rad={}".format(curve),
                },
            )

        current_node = snapshot.get("current_node_id")
        for node in nodes:
            node_id = node["id"]
            position = positions[node_id]
            status = node.get("room_status", "unexplored")
            color = self.STATUS_COLORS.get(status, "#95a5a6")
            edge_color = "#0066cc" if node_id == current_node else "#263238"
            line_width = 3.0 if node_id == current_node else 1.2
            axis.scatter(
                [position[0]],
                [position[1]],
                s=1300,
                color=color,
                edgecolors=edge_color,
                linewidths=line_width,
                zorder=3,
            )
            axis.text(
                position[0],
                position[1],
                "R{}".format(node_id),
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
                zorder=4,
            )
            axis.text(
                position[0],
                position[1] - 0.09,
                status,
                ha="center",
                va="top",
                color="#263238",
                fontsize=8,
            )
        axis.text(
            0.01,
            0.01,
            "rooms {}   directed edges {}   current {}".format(
                snapshot.get("room_count", 0),
                snapshot.get("edge_count", 0),
                "R{}".format(current_node) if current_node is not None else "none",
            ),
            transform=axis.transAxes,
            fontsize=9,
            color="#263238",
        )

    def _draw_map(
        self,
        occupied,
        explored,
        agent_xy,
        heading_degrees,
        goal_xy,
        doors,
        raw_doors,
        frontiers,
        trajectory_xy,
    ):
        axis = self.map_axis
        axis.clear()
        axis.set_title("Online map and decisions")
        axis.imshow(self._map_image(occupied, explored), interpolation="nearest")

        if trajectory_xy:
            trajectory = np.asarray(trajectory_xy, dtype=np.float32)
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="#1976d2",
                linewidth=1.2,
                alpha=0.8,
            )
        for door in doors:
            start = door.get("start")
            end = door.get("end")
            if start is None or end is None:
                continue
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#d81b60",
                linewidth=2.5,
            )
        if raw_doors:
            raw = np.asarray(raw_doors)
            if raw.ndim == 2 and raw.shape[1] >= 2:
                axis.scatter(raw[:, 0], raw[:, 1], s=8, color="#f48fb1")
        if frontiers:
            frontier_array = np.asarray(frontiers)
            if frontier_array.ndim == 2 and frontier_array.shape[1] >= 2:
                axis.scatter(
                    frontier_array[:, 1],
                    frontier_array[:, 0],
                    s=24,
                    color="#ef6c00",
                    marker="x",
                )
        if goal_xy is not None:
            axis.scatter(
                [goal_xy[0]],
                [goal_xy[1]],
                s=90,
                color="#76ff03",
                edgecolors="#33691e",
                linewidths=1.0,
                marker="*",
                zorder=5,
            )
        if agent_xy is not None:
            heading = np.deg2rad(float(heading_degrees))
            dx = np.sin(heading)
            dy = np.cos(heading)
            axis.scatter(
                [agent_xy[0]],
                [agent_xy[1]],
                s=75,
                color="#00b8d4",
                edgecolors="#004d40",
                zorder=6,
            )
            axis.arrow(
                agent_xy[0],
                agent_xy[1],
                dx * 14.0,
                dy * 14.0,
                width=1.2,
                head_width=7.0,
                head_length=7.0,
                color="#e53935",
                length_includes_head=True,
                zorder=6,
            )
        axis.set_xticks([])
        axis.set_yticks([])

    def _draw_metrics(self, coverage_history, step, explored_ratio, explored_area):
        axis = self.metrics_axis
        axis.clear()
        axis.set_title("Exploration progress")
        if coverage_history:
            axis.plot(
                range(1, len(coverage_history) + 1),
                coverage_history,
                color="#00897b",
                linewidth=1.8,
            )
        axis.set_ylim(0.0, 1.0)
        axis.set_xlim(1, max(2, len(coverage_history)))
        axis.grid(True, color="#dfe3e6", linewidth=0.7)
        axis.set_ylabel("ratio")
        axis.text(
            0.99,
            0.06,
            "step {}   ratio {:.3f}   area {:.2f} m2".format(
                int(step),
                float(explored_ratio),
                float(explored_area),
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#263238",
        )

    def _draw_events(self, status, events):
        axis = self.events_axis
        axis.clear()
        axis.axis("off")
        axis.set_title("Runtime and topology events", loc="left")
        lines = [
            "phase: {}    action: {}".format(
                status.get("phase", "unknown"),
                status.get("action", "unknown"),
            ),
            "scene: {}".format(status.get("scene_name") or "initializing"),
            "doors: {} accepted / {} raw    frontiers: {}".format(
                status.get("door_count", 0),
                status.get("raw_door_count", 0),
                status.get("frontier_count", 0),
            ),
        ]
        for event in events[-7:]:
            lines.append(
                "#{:03d} s{:04d} {}".format(
                    int(event.get("sequence", 0)),
                    int(event.get("step", 0)),
                    event.get("event_type", "unknown"),
                )
            )
        axis.text(
            0.0,
            0.96,
            "\n".join(lines),
            transform=axis.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=9,
            color="#263238",
            linespacing=1.35,
        )

    def render(
        self,
        *,
        step,
        rgb,
        occupied,
        explored,
        agent_xy,
        heading_degrees,
        goal_xy,
        doors,
        raw_doors,
        frontiers,
        trajectory_xy,
        topology_snapshot,
        coverage_history,
        explored_ratio,
        explored_area,
        status,
        events,
    ):
        self.last_render_step = int(step)
        self.rgb_axis.clear()
        self.rgb_axis.set_title("RGB and door detector")
        self.rgb_axis.imshow(np.asarray(rgb))
        self.rgb_axis.set_xticks([])
        self.rgb_axis.set_yticks([])
        self._draw_map(
            occupied,
            explored,
            agent_xy,
            heading_degrees,
            goal_xy,
            doors,
            raw_doors,
            frontiers,
            trajectory_xy,
        )
        self._draw_topology(topology_snapshot)
        self._draw_metrics(
            coverage_history,
            step,
            explored_ratio,
            explored_area,
        )
        self._draw_events(status, events)
        self.figure.suptitle(
            "Active Room Segmentation | step {:04d} | {}".format(
                int(step),
                status.get("phase", "unknown"),
            ),
            fontsize=14,
            fontweight="bold",
        )
        self._draw_capture_marker(step)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

        if (
            self.frame_every_steps > 0
            and step > 0
            and step % self.frame_every_steps == 0
            and step != self.last_saved_step
        ):
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = self.frame_dir / "frame_{:06d}.png".format(int(step))
            self._save_fixed_render(frame_path, self.FRAME_DPI)
            self.frame_paths.append(frame_path)
            self.last_saved_step = int(step)
        if self.refresh_seconds > 0.0:
            from matplotlib import pyplot as plt

            plt.pause(self.refresh_seconds)

    def finalize(self, topology_snapshot, event_summary):
        final_path = self.run_dir / "visualization_final.png"
        self._save_fixed_render(final_path, self.FINAL_DPI)
        manifest = {
            "status": "completed",
            "last_render_step": self.last_render_step,
            "frame_every_steps": self.frame_every_steps,
            "frame_count": len(self.frame_paths),
            "frames": [path.relative_to(self.run_dir).as_posix() for path in self.frame_paths],
            "final_image": final_path.name,
            "final_image_sha256": _sha256(final_path),
            "capture_marker_scheme": "sha256_run_step_v1",
            "capture_identity_sha256": hashlib.sha256(
                self.capture_identity.encode("utf-8")
            ).hexdigest(),
            "frame_size": [
                int(self.FIXED_RENDER_SIZE_INCHES[0] * self.FRAME_DPI),
                int(self.FIXED_RENDER_SIZE_INCHES[1] * self.FRAME_DPI),
            ],
            "final_image_size": [
                int(self.FIXED_RENDER_SIZE_INCHES[0] * self.FINAL_DPI),
                int(self.FIXED_RENDER_SIZE_INCHES[1] * self.FINAL_DPI),
            ],
            "topology": topology_snapshot,
            "events": event_summary,
        }
        _write_json_atomic(self.run_dir / "visualization_manifest.json", manifest)
        return manifest
