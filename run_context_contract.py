import hashlib
import json
from collections.abc import Mapping


STRICT_VISUAL_CAPTURE_STEPS = (
    ("first", 5),
    ("later", 10),
    ("mid", 30),
)
STRICT_LIVE_FIGURE_SIZE_INCHES = (32.0, 18.0)
STRICT_LIVE_FIGURE_DPI = 100
STRICT_LIVE_CANVAS_SIZE = tuple(
    int(size * STRICT_LIVE_FIGURE_DPI)
    for size in STRICT_LIVE_FIGURE_SIZE_INCHES
)
STRICT_CAPTURE_MARKER_SYNC = (1, 0, 1, 1, 0, 1, 0, 0)
STRICT_CAPTURE_MARKER_PAYLOAD_BITS = 64
STRICT_CAPTURE_MARKER_X = 0.37
STRICT_CAPTURE_MARKER_Y = 0.012
STRICT_CAPTURE_MARKER_WIDTH = 0.26
STRICT_CAPTURE_MARKER_HEIGHT = 0.018


def strict_capture_marker_bits(run_id, step):
    run_id = str(run_id)
    step = int(step)
    if not run_id or step < 0:
        raise ValueError("Capture marker requires a run ID and non-negative step")
    digest = hashlib.sha256(
        "{}:{}".format(run_id, step).encode("utf-8")
    ).digest()
    payload = tuple(
        (digest[bit_index // 8] >> (7 - bit_index % 8)) & 1
        for bit_index in range(STRICT_CAPTURE_MARKER_PAYLOAD_BITS)
    )
    return STRICT_CAPTURE_MARKER_SYNC + payload


def _field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _json_value(value):
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("Episode contract contains a non-JSON value: {!r}".format(value))


def _float_vector(value, name):
    if not isinstance(value, (list, tuple)):
        raise TypeError("{} must be a list or tuple".format(name))
    return [float(item) for item in value]


def episode_contract(episode):
    goals = []
    for goal in _field(episode, "goals", []):
        radius = _field(goal, "radius")
        goals.append(
            {
                "position": _float_vector(
                    _field(goal, "position"),
                    "goal.position",
                ),
                "radius": None if radius is None else float(radius),
            }
        )
    if not goals:
        raise RuntimeError("Episode contract must contain at least one goal")
    return {
        "episode_id": str(_field(episode, "episode_id")),
        "scene_id": str(_field(episode, "scene_id")),
        "start_position": _float_vector(
            _field(episode, "start_position"),
            "start_position",
        ),
        "start_rotation": _float_vector(
            _field(episode, "start_rotation"),
            "start_rotation",
        ),
        "goals": goals,
        "info": _json_value(_field(episode, "info", {})),
        "start_room": _json_value(_field(episode, "start_room")),
        "shortest_paths": _json_value(_field(episode, "shortest_paths")),
    }


def episode_contract_sha256(episode):
    payload = json.dumps(
        episode_contract(episode),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
