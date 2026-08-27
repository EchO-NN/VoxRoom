from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid


_SCHEMA_VERSION = 1
_DISPLAY_PRIMVARS = ("displayColor", "displayOpacity")
_NONCONSTANT_INTERPOLATIONS = {"faceVarying", "uniform", "varying", "vertex"}


@dataclass(frozen=True)
class DisplayPrimvarOverlayResult:
    source_path: Path
    effective_path: Path
    correction_count: int
    covered_values: int
    cache_hit: bool


@dataclass(frozen=True)
class _Correction:
    prim_path: str
    attribute_name: str
    type_name: object
    expected_values: int


def _file_state(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _layer_states(stage, source_path: Path) -> list[dict]:
    paths = {source_path.resolve()}
    for layer in stage.GetUsedLayers():
        raw_path = str(getattr(layer, "realPath", "") or "").strip()
        if raw_path:
            path = Path(raw_path).expanduser()
            if path.is_file():
                paths.add(path.resolve())
    return [_file_state(path) for path in sorted(paths, key=str)]


def _manifest_is_current(manifest: dict, source_path: Path) -> bool:
    if int(manifest.get("schema_version", -1)) != _SCHEMA_VERSION:
        return False
    if str(manifest.get("source_path", "")) != str(source_path):
        return False
    states = manifest.get("layer_states")
    if not isinstance(states, list) or not states:
        return False
    try:
        return all(_file_state(Path(item["path"])) == item for item in states)
    except (KeyError, OSError, TypeError):
        return False


def _expected_value_count(mesh, interpolation: str) -> int:
    if interpolation == "faceVarying":
        value = mesh.GetFaceVertexIndicesAttr().Get()
    elif interpolation == "uniform":
        value = mesh.GetFaceVertexCountsAttr().Get()
    elif interpolation in {"vertex", "varying"}:
        value = mesh.GetPointsAttr().Get()
    else:
        return 1
    return 0 if value is None else len(value)


def _find_corrections(stage) -> list[_Correction]:
    from pxr import UsdGeom

    corrections: list[_Correction] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        primvars = UsdGeom.PrimvarsAPI(prim)
        expected_by_interpolation: dict[str, int] = {}
        for name in _DISPLAY_PRIMVARS:
            primvar = primvars.GetPrimvar(name)
            if not primvar or not primvar.IsDefined() or primvar.IsIndexed():
                continue
            interpolation = str(primvar.GetInterpolation())
            if interpolation not in _NONCONSTANT_INTERPOLATIONS:
                continue
            values = primvar.GetAttr().Get()
            if values is None or len(values) != 1:
                continue
            expected = expected_by_interpolation.get(interpolation)
            if expected is None:
                expected = _expected_value_count(mesh, interpolation)
                expected_by_interpolation[interpolation] = expected
            if expected <= 1:
                continue
            attribute = primvar.GetAttr()
            corrections.append(
                _Correction(
                    prim_path=str(prim.GetPath()),
                    attribute_name=str(attribute.GetName()),
                    type_name=attribute.GetTypeName(),
                    expected_values=int(expected),
                )
            )
    return corrections


def _write_overlay(source_path: Path, output_path: Path, corrections: list[_Correction]) -> None:
    from pxr import Sdf, Usd, UsdGeom

    temporary_path = output_path.with_name(
        "%s.%s.tmp.usda" % (output_path.stem, uuid.uuid4().hex)
    )
    layer = Sdf.Layer.CreateNew(str(temporary_path))
    if layer is None:
        raise RuntimeError("failed to create USD overlay %s" % temporary_path)
    layer.subLayerPaths = [str(source_path)]
    layer.customLayerData = {
        "voxroom:sanitizer": "display_primvar_singleton_v1",
        "voxroom:source": str(source_path),
        "voxroom:correctionCount": int(len(corrections)),
    }
    stage = Usd.Stage.Open(layer)
    if stage is None:
        raise RuntimeError("failed to open USD overlay %s" % temporary_path)
    stage.SetEditTarget(layer)
    for correction in corrections:
        prim = stage.OverridePrim(correction.prim_path)
        attribute = prim.CreateAttribute(correction.attribute_name, correction.type_name)
        primvar = UsdGeom.Primvar(attribute)
        if not primvar.SetInterpolation(UsdGeom.Tokens.constant):
            raise RuntimeError(
                "failed to set constant interpolation on %s.%s"
                % (correction.prim_path, correction.attribute_name)
            )
    if not layer.Save():
        raise RuntimeError("failed to save USD overlay %s" % temporary_path)
    os.replace(temporary_path, output_path)


def _write_manifest(path: Path, payload: dict) -> None:
    temporary_path = path.with_name("%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def prepare_display_primvar_overlay(
    source_usd: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
) -> DisplayPrimvarOverlayResult:
    """Return a scene path whose singleton display primvars are declared constant.

    InteriorAgent scenes contain non-indexed one-value display primvars declared
    as face-varying. Hydra treats those declarations as corrupted topology-sized
    buffers. The generated USD layer only corrects interpolation metadata; it
    sublayers the original scene and leaves all geometry and authored values intact.
    """

    from pxr import Usd

    source_path = Path(source_usd).expanduser().resolve(strict=True)
    cache_path = Path(cache_root).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    source_key = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]
    stem = "%s-%s-display-primvars-v1" % (source_path.stem, source_key)
    overlay_path = cache_path / (stem + ".usda")
    manifest_path = cache_path / (stem + ".json")
    lock_path = cache_path / (stem + ".lock")

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            if _manifest_is_current(manifest, source_path):
                correction_count = int(manifest.get("correction_count", 0))
                effective_path = overlay_path if correction_count else source_path
                if effective_path.is_file():
                    return DisplayPrimvarOverlayResult(
                        source_path=source_path,
                        effective_path=effective_path,
                        correction_count=correction_count,
                        covered_values=int(manifest.get("covered_values", 0)),
                        cache_hit=True,
                    )

        source_stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
        if source_stage is None:
            raise RuntimeError("failed to inspect USD scene %s" % source_path)
        corrections = _find_corrections(source_stage)
        layer_states = _layer_states(source_stage, source_path)
        covered_values = sum(item.expected_values for item in corrections)
        if corrections:
            _write_overlay(source_path, overlay_path, corrections)
            effective_path = overlay_path
        else:
            overlay_path.unlink(missing_ok=True)
            effective_path = source_path
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "source_path": str(source_path),
            "effective_path": str(effective_path),
            "correction_count": int(len(corrections)),
            "covered_values": int(covered_values),
            "layer_states": layer_states,
        }
        _write_manifest(manifest_path, manifest)
        return DisplayPrimvarOverlayResult(
            source_path=source_path,
            effective_path=effective_path,
            correction_count=len(corrections),
            covered_values=covered_values,
            cache_hit=False,
        )
