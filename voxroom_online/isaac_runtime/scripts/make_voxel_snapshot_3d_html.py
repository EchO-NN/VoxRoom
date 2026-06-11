from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a self-contained WebGL 3D voxel snapshot viewer.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--max-occupied", type=int, default=260000)
    parser.add_argument("--max-free", type=int, default=120000)
    parser.add_argument("--max-floor", type=int, default=180000)
    parser.add_argument("--overlay-z-step-m", type=float, default=0.15)
    parser.add_argument("--overlay-z-max-m", type=float, default=2.4)
    parser.add_argument("--crop-rc", nargs=4, type=int, metavar=("R0", "R1", "C0", "C1"), default=None)
    parser.add_argument("--title-suffix", default="")
    parser.add_argument("--hide-overlays", action="store_true")
    parser.add_argument("--only-occupied", action="store_true")
    args = parser.parse_args(argv)

    snapshot = Path(args.snapshot).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(snapshot, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}

    state = np.asarray(arrays["voxel_occupancy_state_zyx"], dtype=np.uint8)
    if state.ndim != 3:
        raise ValueError("voxel_occupancy_state_zyx must be [Z,H,W]")
    z_centers = np.asarray(arrays.get("voxel_occupancy_z_centers_m", np.arange(state.shape[0]) * float(args.resolution_m)), dtype=np.float32)
    if z_centers.shape[0] != state.shape[0]:
        z_centers = np.arange(state.shape[0], dtype=np.float32) * float(args.resolution_m)
    res_m = float(args.resolution_m)
    shape_hw = (int(state.shape[1]), int(state.shape[2]))
    center_r = (shape_hw[0] - 1) * 0.5
    center_c = (shape_hw[1] - 1) * 0.5
    xy_keep = np.ones(shape_hw, dtype=bool)
    crop_bounds: tuple[int, int, int, int] | None = None
    if args.crop_rc is not None:
        r0, r1, c0, c1 = [int(v) for v in args.crop_rc]
        r0 = max(0, min(shape_hw[0], r0))
        r1 = max(0, min(shape_hw[0], r1))
        c0 = max(0, min(shape_hw[1], c0))
        c1 = max(0, min(shape_hw[1], c1))
        if r1 <= r0 or c1 <= c0:
            raise ValueError("--crop-rc must satisfy R1 > R0 and C1 > C0 after clipping")
        xy_keep[:, :] = False
        xy_keep[r0:r1, c0:c1] = True
        crop_bounds = (r0, r1, c0, c1)
        crop_center_r = (r0 + r1 - 1) * 0.5
        crop_center_c = (c0 + c1 - 1) * 0.5
        view_center_y = (crop_center_r - center_r) * res_m
        view_center_x = (crop_center_c - center_c) * res_m
        scene_radius_m = float(max(r1 - r0, c1 - c0) * res_m * 0.78)
    else:
        view_center_x = 0.0
        view_center_y = 0.0
        scene_radius_m = float(max(shape_hw) * res_m * 0.72)

    layers: list[dict[str, object]] = []
    occupied_mask = (state == 2) & xy_keep[None, :, :]
    free_mask = (state == 1) & xy_keep[None, :, :]
    occupied_pos = _voxel_positions(occupied_mask, z_centers, res_m, center_r, center_c, max_points=int(args.max_occupied))
    if occupied_pos.size:
        layers.append(_uniform_layer("occupied_voxels", occupied_pos, (255, 70, 45, 230), 3.8, True))
    if not bool(args.only_occupied):
        free_pos = _voxel_positions(free_mask, z_centers, res_m, center_r, center_c, max_points=int(args.max_free))
        if free_pos.size:
            layers.append(_uniform_layer("sampled_free_voxels", free_pos, (100, 210, 255, 42), 2.0, False))

    label_map = np.asarray(arrays.get("voxel_final_room_label_map", arrays.get("final_room_label_map", np.zeros(shape_hw, dtype=np.int32))), dtype=np.int32)
    label_map = np.where(xy_keep, label_map, 0)
    if not bool(args.only_occupied):
        floor_pos, floor_color = _floor_label_points(label_map, res_m, center_r, center_c, max_points=int(args.max_floor))
        if floor_pos.size:
            layers.append(_colored_layer("room_floor_labels", floor_pos, floor_color, 3.0, True))

    overlay_z = np.arange(0.08, float(args.overlay_z_max_m) + 1e-6, max(float(args.overlay_z_step_m), 0.05), dtype=np.float32)
    wall = _mask_union(
        arrays,
        shape_hw,
        (
            "voxel_wall_xy",
            "voxel_wall_after_step1_map",
            "voxel_final_separator_map",
            "accepted_separators",
        ),
    ) & xy_keep
    wall_extension = _mask_union(
        arrays,
        shape_hw,
        (
            "voxel_step1_wall_gap_fill_map",
            "voxel_step2_extension_candidate_map",
            "voxel_step2_extension_separator_map",
            "wall_extension_boundary_mask",
        ),
    ) & xy_keep
    door_seed = _mask_union(
        arrays,
        shape_hw,
        (
            "voxel_door_raw_seed_mask",
            "voxel_door_seed_mask",
            "voxel_door_seed_line_primitive_mask",
            "voxel_door_extensible_primitive_mask",
        ),
    ) & xy_keep
    door_seed_extension = _mask_union(
        arrays,
        shape_hw,
        (
            "voxel_door_extension_trials_map",
            "voxel_door_centerline_mask",
            "voxel_door_cut_mask",
            "voxel_door_partition_cut_candidate_mask",
            "voxel_door_partition_cut_mask",
            "voxel_door_topology_effective_cut_mask",
            "voxel_current_door_cut_mask",
            "voxel_current_door_topology_effective_mask",
            "voxel_stable_door_cut_mask",
            "voxel_door_stable_cut_mask",
            "partial_door_extension_cut_mask",
            "door_completion_boundary_mask",
        ),
    ) & xy_keep
    overlay_defs = (
        ("wall_lines", wall, (255, 35, 35, 255), 6.5, not bool(args.hide_overlays)),
        ("wall_extensions", wall_extension, (190, 70, 255, 255), 6.5, not bool(args.hide_overlays)),
        ("door_seed_extensions", door_seed_extension, (35, 230, 85, 255), 7.5, not bool(args.hide_overlays)),
        ("door_seeds", door_seed, (45, 120, 255, 255), 8.5, not bool(args.hide_overlays)),
    )
    if not bool(args.only_occupied):
        for name, mask, color, size, visible in overlay_defs:
            pos = _xy_mask_columns(mask, overlay_z, res_m, center_r, center_c)
            if pos.size:
                layers.append(_uniform_layer(name, pos, color, size, visible))

    html = _render_html(
        title=f"{snapshot.parent.parent.name} {snapshot.stem} voxel 3D{(' ' + args.title_suffix) if args.title_suffix else ''}",
        snapshot=str(snapshot),
        layers=layers,
        x_center=float(view_center_x),
        y_center=float(view_center_y),
        z_center=float(np.nanmean(z_centers)) if z_centers.size else 1.6,
        scene_radius_m=max(1.0, float(scene_radius_m)),
        counts={
            "occupied_voxels": int(np.count_nonzero(occupied_mask)),
            "free_voxels": int(np.count_nonzero(free_mask)),
            "unknown_voxels": int(np.count_nonzero((state == 0) & xy_keep[None, :, :])),
            "wall_cells": int(np.count_nonzero(wall)),
            "wall_extension_cells": int(np.count_nonzero(wall_extension)),
            "door_seed_cells": int(np.count_nonzero(door_seed)),
            "door_seed_extension_cells": int(np.count_nonzero(door_seed_extension)),
            "crop_r0": 0 if crop_bounds is None else int(crop_bounds[0]),
            "crop_r1": shape_hw[0] if crop_bounds is None else int(crop_bounds[1]),
            "crop_c0": 0 if crop_bounds is None else int(crop_bounds[2]),
            "crop_c1": shape_hw[1] if crop_bounds is None else int(crop_bounds[3]),
        },
    )
    out_path.write_text(html, encoding="utf-8")
    print(json.dumps({"out": str(out_path), "layers": len(layers), "points": int(sum(int(layer["count"]) for layer in layers))}, ensure_ascii=False))
    return 0


def _voxel_positions(mask_zyx: np.ndarray, z_centers: np.ndarray, res_m: float, center_r: float, center_c: float, *, max_points: int) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask_zyx, dtype=bool))
    coords = _subsample_rows(coords, max_points=max_points)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = z_centers[coords[:, 0]].astype(np.float32)
    y = (coords[:, 1].astype(np.float32) - float(center_r)) * float(res_m)
    x = (coords[:, 2].astype(np.float32) - float(center_c)) * float(res_m)
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


def _floor_label_points(label_map: np.ndarray, res_m: float, center_r: float, center_c: float, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(np.asarray(label_map, dtype=np.int32) > 0)
    coords = _subsample_rows(coords, max_points=max_points)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    labels = np.asarray(label_map, dtype=np.int32)[coords[:, 0], coords[:, 1]]
    y = (coords[:, 0].astype(np.float32) - float(center_r)) * float(res_m)
    x = (coords[:, 1].astype(np.float32) - float(center_c)) * float(res_m)
    pos = np.column_stack((x, y, np.full(coords.shape[0], -0.035, dtype=np.float32))).astype(np.float32, copy=False)
    colors = np.asarray([_label_rgba(int(label), alpha=0.82) for label in labels], dtype=np.float32)
    return pos, colors


def _xy_mask_columns(mask: np.ndarray, z_values: np.ndarray, res_m: float, center_r: float, center_c: float) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0 or z_values.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    repeated = np.repeat(coords, int(z_values.shape[0]), axis=0)
    z = np.tile(z_values, int(coords.shape[0]))
    y = (repeated[:, 0].astype(np.float32) - float(center_r)) * float(res_m)
    x = (repeated[:, 1].astype(np.float32) - float(center_c)) * float(res_m)
    return np.column_stack((x, y, z.astype(np.float32))).astype(np.float32, copy=False)


def _mask_union(arrays: Mapping[str, np.ndarray], shape: tuple[int, int], keys: Sequence[str]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for key in keys:
        value = arrays.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape == shape:
            out |= np.asarray(arr, dtype=bool)
    return out


def _subsample_rows(coords: np.ndarray, *, max_points: int) -> np.ndarray:
    if int(max_points) <= 0 or coords.shape[0] <= int(max_points):
        return coords
    idx = np.linspace(0, coords.shape[0] - 1, int(max_points), dtype=np.int64)
    return coords[idx]


def _uniform_layer(name: str, positions: np.ndarray, rgba: tuple[int, int, int, int], point_size: float, visible: bool) -> dict[str, object]:
    color = [float(v) / 255.0 for v in rgba]
    return {
        "name": name,
        "count": int(positions.shape[0]),
        "positions": _b64_float32(positions),
        "color": color,
        "pointSize": float(point_size),
        "visible": bool(visible),
    }


def _colored_layer(name: str, positions: np.ndarray, colors: np.ndarray, point_size: float, visible: bool) -> dict[str, object]:
    return {
        "name": name,
        "count": int(positions.shape[0]),
        "positions": _b64_float32(positions),
        "colors": _b64_float32(colors),
        "pointSize": float(point_size),
        "visible": bool(visible),
    }


def _b64_float32(array: np.ndarray) -> str:
    arr = np.asarray(array, dtype="<f4")
    return base64.b64encode(arr.tobytes(order="C")).decode("ascii")


def _label_rgba(label: int, *, alpha: float) -> tuple[float, float, float, float]:
    return (
        (80 + ((int(label) * 53) % 140)) / 255.0,
        (90 + ((int(label) * 97) % 125)) / 255.0,
        (105 + ((int(label) * 193) % 115)) / 255.0,
        float(alpha),
    )


def _render_html(
    *,
    title: str,
    snapshot: str,
    layers: list[dict[str, object]],
    x_center: float,
    y_center: float,
    z_center: float,
    scene_radius_m: float,
    counts: Mapping[str, int],
) -> str:
    payload = {
        "title": title,
        "snapshot": snapshot,
        "layers": layers,
        "xCenter": float(x_center),
        "yCenter": float(y_center),
        "zCenter": float(z_center),
        "sceneRadiusM": float(scene_radius_m),
        "counts": dict(counts),
    }
    return """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>""" + _html_escape(title) + """</title>
<style>
html,body{margin:0;height:100%;overflow:hidden;background:#0f1117;color:#e7e7ea;font-family:Arial,sans-serif}
#gl{position:fixed;inset:0;width:100%;height:100%;display:block}
#panel{position:fixed;top:12px;left:12px;background:rgba(10,12,17,.84);border:1px solid #343844;border-radius:6px;padding:10px 12px;min-width:290px;font-size:13px;line-height:1.35}
#panel h1{font-size:14px;margin:0 0 6px 0;font-weight:600}
#panel label{display:block;margin:4px 0;white-space:nowrap}
#panel input[type=range]{width:140px;vertical-align:middle}
#hint{position:fixed;right:12px;bottom:10px;color:#c8cad2;background:rgba(10,12,17,.74);border:1px solid #343844;border-radius:6px;padding:7px 9px;font-size:12px}
button{background:#222735;border:1px solid #52596b;color:#f0f0f2;border-radius:4px;padding:3px 8px;margin-top:5px}
.count{color:#aeb4c4;font-size:12px;margin-top:6px;max-width:520px}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="panel">
<h1>""" + _html_escape(title) + """</h1>
<div id="layers"></div>
<label>z scale <input id="zScale" type="range" min="1" max="6" step="0.1" value="2.5"> <span id="zText">2.5</span></label>
<label>point scale <input id="pScale" type="range" min="0.4" max="3.0" step="0.1" value="1.0"> <span id="pText">1.0</span></label>
<button id="reset">reset view</button>
<div class="count" id="counts"></div>
</div>
<div id="hint">drag: rotate | right/shift drag: pan | wheel: zoom | WASD/arrows/QE: move</div>
<script>
const DATA = """ + json.dumps(payload, separators=(",", ":")) + """;

function decodeF32(b64){
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl', {alpha:false, antialias:true});
if(!gl) alert('WebGL is not available');

const vs = `
attribute vec3 a_position;
attribute vec4 a_color;
uniform mat4 u_matrix;
uniform float u_pointSize;
uniform vec4 u_uniformColor;
uniform bool u_useColorBuffer;
varying vec4 v_color;
void main(){
  gl_Position = u_matrix * vec4(a_position, 1.0);
  gl_PointSize = u_pointSize;
  v_color = u_useColorBuffer ? a_color : u_uniformColor;
}`;
const fs = `
precision mediump float;
varying vec4 v_color;
void main(){
  vec2 p = gl_PointCoord - vec2(0.5, 0.5);
  if(dot(p,p) > 0.25) discard;
  gl_FragColor = v_color;
}`;
function shader(type, source){
  const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
  if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const program = gl.createProgram();
gl.attachShader(program, shader(gl.VERTEX_SHADER, vs));
gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(program);
if(!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);
const loc = {
  position: gl.getAttribLocation(program, 'a_position'),
  color: gl.getAttribLocation(program, 'a_color'),
  matrix: gl.getUniformLocation(program, 'u_matrix'),
  pointSize: gl.getUniformLocation(program, 'u_pointSize'),
  uniformColor: gl.getUniformLocation(program, 'u_uniformColor'),
  useColorBuffer: gl.getUniformLocation(program, 'u_useColorBuffer')
};

const layers = DATA.layers.map((layer, idx) => {
  const pos = decodeF32(layer.positions);
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  let colorBuf = null;
  if(layer.colors){
    colorBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, decodeF32(layer.colors), gl.STATIC_DRAW);
  }
  return {...layer, idx, posBuf, colorBuf, visible: !!layer.visible};
});

const layerBox = document.getElementById('layers');
for(const layer of layers){
  const label = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.checked = layer.visible;
  cb.onchange = () => { layer.visible = cb.checked; draw(); };
  label.appendChild(cb);
  label.appendChild(document.createTextNode(' ' + layer.name + ' (' + layer.count + ')'));
  layerBox.appendChild(label);
}
document.getElementById('counts').textContent = Object.entries(DATA.counts).map(([k,v])=>k+'='+v).join('  ');

let yaw = -0.7, pitch = 1.05, dist = Math.max(12, DATA.sceneRadiusM * 1.55), panX = 0, panY = 0, panZ = 0;
let dragging = false, lastX = 0, lastY = 0, panMode = false;
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; panMode = e.button === 2 || e.shiftKey; });
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {
  if(!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY; lastX = e.clientX; lastY = e.clientY;
  if(panMode){
    const panSpeed = Math.max(0.018, dist * 0.0018);
    panX += dx * panSpeed;
    panY -= dy * panSpeed;
  }
  else {
    yaw += dx * 0.008;
    pitch += dy * 0.008;
  }
  draw();
});
canvas.addEventListener('wheel', e => { e.preventDefault(); dist *= Math.exp(e.deltaY * 0.001); dist = Math.max(0.5, Math.min(1200, dist)); draw(); }, {passive:false});
window.addEventListener('keydown', e => {
  const key = e.key.toLowerCase();
  const step = Math.max(0.08, dist * (e.shiftKey ? 0.035 : 0.012));
  let used = true;
  if(key === 'a' || key === 'arrowleft') panX += step;
  else if(key === 'd' || key === 'arrowright') panX -= step;
  else if(key === 'w' || key === 'arrowup') panY -= step;
  else if(key === 's' || key === 'arrowdown') panY += step;
  else if(key === 'q') panZ -= step;
  else if(key === 'e') panZ += step;
  else used = false;
  if(used){ e.preventDefault(); draw(); }
});
document.getElementById('zScale').oninput = e => { document.getElementById('zText').textContent = e.target.value; draw(); };
document.getElementById('pScale').oninput = e => { document.getElementById('pText').textContent = e.target.value; draw(); };
document.getElementById('reset').onclick = () => { yaw=-0.7; pitch=1.05; dist=Math.max(12, DATA.sceneRadiusM*1.55); panX=0; panY=0; panZ=0; draw(); };

function resize(){
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if(canvas.width !== w || canvas.height !== h){ canvas.width = w; canvas.height = h; gl.viewport(0,0,w,h); }
}
window.addEventListener('resize', draw);

function m4mul(a,b){
  const o = new Float32Array(16);
  for(let c=0;c<4;c++) for(let r=0;r<4;r++) o[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
  return o;
}
function perspective(fovy, aspect, near, far){
  const f = 1/Math.tan(fovy/2), nf = 1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function trans(x,y,z){ return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]); }
function scale(x,y,z){ return new Float32Array([x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1]); }
function rotX(a){ const c=Math.cos(a), s=Math.sin(a); return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]); }
function rotZ(a){ const c=Math.cos(a), s=Math.sin(a); return new Float32Array([c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1]); }
function matrix(){
  const aspect = canvas.width / Math.max(1, canvas.height);
  const zScale = parseFloat(document.getElementById('zScale').value);
  let m = perspective(Math.PI/4, aspect, 0.05, 500);
  m = m4mul(m, trans(panX, panY, -dist + panZ));
  m = m4mul(m, rotX(pitch));
  m = m4mul(m, rotZ(yaw));
  m = m4mul(m, scale(1,1,zScale));
  m = m4mul(m, trans(-DATA.xCenter,-DATA.yCenter,-DATA.zCenter));
  return m;
}
function draw(){
  resize();
  gl.clearColor(0.055,0.063,0.086,1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.uniformMatrix4fv(loc.matrix, false, matrix());
  const pScale = parseFloat(document.getElementById('pScale').value);
  for(const layer of layers){
    if(!layer.visible || layer.count <= 0) continue;
    gl.bindBuffer(gl.ARRAY_BUFFER, layer.posBuf);
    gl.enableVertexAttribArray(loc.position);
    gl.vertexAttribPointer(loc.position, 3, gl.FLOAT, false, 0, 0);
    if(layer.colorBuf){
      gl.uniform1i(loc.useColorBuffer, 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, layer.colorBuf);
      gl.enableVertexAttribArray(loc.color);
      gl.vertexAttribPointer(loc.color, 4, gl.FLOAT, false, 0, 0);
    } else {
      gl.uniform1i(loc.useColorBuffer, 0);
      gl.disableVertexAttribArray(loc.color);
      gl.vertexAttrib4f(loc.color, 1, 1, 1, 1);
      gl.uniform4fv(loc.uniformColor, layer.color);
    }
    gl.uniform1f(loc.pointSize, layer.pointSize * pScale);
    gl.drawArrays(gl.POINTS, 0, layer.count);
  }
}
draw();
</script>
</body>
</html>
"""


def _html_escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    raise SystemExit(main())
