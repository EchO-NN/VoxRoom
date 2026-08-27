from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from voxroom_online.isaac_runtime.comparison.validate_all_baselines import validate_all_baselines
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_index import load_index


DEFAULT_METHODS = ("voxroom", "topology_visual_active", "dude_incremental", "rose2", "morphological", "distance_transform", "voronoi")


def validate_paper_ready(
    *,
    run_root: Path,
    eval_root: Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    require_csr: bool = True,
    require_native: bool = True,
    require_original_commits: bool = True,
    require_no_placeholder: bool = True,
    require_environment_lock: bool = True,
    require_repos_lock: bool = True,
    require_strict_replay_manifest: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root)
    eval_root = Path(eval_root)
    problems: list[str] = []
    method_rows: dict[str, dict[str, Any]] = {}
    fallback_detected = False
    placeholder_detected = False
    missing_gt = False
    missing_csr = False

    if require_environment_lock:
        problems.extend(_environment_lock_problems(run_root))
    if require_repos_lock:
        problems.extend(_repos_lock_problems(run_root))
    if require_strict_replay_manifest:
        problems.extend(_baseline_manifest_problems(run_root))

    for method in methods:
        index_path = _index_path(eval_root, str(method))
        row: dict[str, Any] = {"snapshots": 0, "gate": "missing_index"}
        if not index_path.exists():
            problems.append("%s missing index: %s" % (method, index_path))
            method_rows[str(method)] = row
            continue
        index = load_index(index_path)
        snapshots = [snap for ep in index.get("episodes", []) for snap in ep.get("snapshots", [])]
        row["snapshots"] = len(snapshots)
        row["gate"] = "pass"
        gt_dir = eval_root / "gt_steps_by_method" / str(method)
        for ep in index.get("episodes", []):
            uid = str(ep["episode_uid"])
            for snap in ep.get("snapshots", []):
                stem = Path(str(snap["snapshot_path"])).stem
                if not (gt_dir / uid / f"{stem}.gt_labels.npy").exists():
                    missing_gt = True
                    row["gate"] = "missing_gt"
                if require_csr and not (eval_root / "csr_by_method" / str(method) / uid / f"{stem}.csr.json").exists():
                    missing_csr = True
                    row["gate"] = "missing_csr"
        metrics_summary = eval_root / "metrics" / str(method) / "summary_metrics.json"
        if not metrics_summary.exists():
            problems.append("%s missing metrics summary: %s" % (method, metrics_summary))
            row["gate"] = "missing_metrics"
        method_rows[str(method)] = row

    baseline_methods = [str(m) for m in methods if str(m) != "voxroom"]
    if require_native and baseline_methods:
        try:
            validate_all_baselines(
                run_root=run_root,
                methods=baseline_methods,
                fail_on_non_main=True,
                require_same_snapshot_set=True,
            )
        except Exception as exc:
            problems.append(str(exc))

    metadata_text = "\n".join(_baseline_metadata_strings(run_root))
    fallback_detected = "fallback" in metadata_text or "smoke" in metadata_text
    placeholder_detected = "placeholder" in metadata_text or "scaffold" in metadata_text or "failure" in metadata_text
    if require_no_placeholder and placeholder_detected:
        problems.append("placeholder/scaffold marker detected in output paths")
    if fallback_detected:
        problems.append("fallback/smoke marker detected in output paths")
    if missing_gt:
        problems.append("missing GT files")
    if require_csr and missing_csr:
        problems.append("missing CSR files")
    if require_original_commits and (eval_root / "reports" / "provenance_manifest.json").exists():
        provenance = _read_json(eval_root / "reports" / "provenance_manifest.json")
        for method in baseline_methods:
            commits = provenance.get("methods", {}).get(method, {}).get("original_repo_commits", "")
            if not commits:
                problems.append("%s missing original repo commit in provenance" % method)

    paper_ready = not problems
    return {
        "paper_ready": bool(paper_ready),
        "run_root": str(run_root),
        "eval_root": str(eval_root),
        "methods": method_rows,
        "fallback_detected": bool(fallback_detected),
        "placeholder_detected": bool(placeholder_detected),
        "missing_gt": bool(missing_gt),
        "missing_csr": bool(missing_csr),
        "problems": problems,
    }


def _index_path(eval_root: Path, method: str) -> Path:
    if method == "voxroom":
        p = Path(eval_root) / "indexes" / "index_voxroom_all.json"
        if p.exists():
            return p
    return Path(eval_root) / "indexes" / f"index_{method}.json"


def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _environment_lock_problems(run_root: Path) -> list[str]:
    path = Path(run_root) / "environment.lock.json"
    if not path.exists():
        return ["missing strict environment lock: %s" % path]
    payload = _read_json(path)
    problems: list[str] = []
    if payload.get("strict_verify") is not True:
        problems.append("environment.lock.json was not produced by STRICT_VERIFY=1")
    if payload.get("ros_python_check", {}).get("ok") is not True:
        problems.append("environment.lock.json ROS python check did not pass")
    if payload.get("topology_checkpoint", {}).get("exists") is not True:
        problems.append("environment.lock.json missing topology checkpoint")
    for name, row in dict(payload.get("external_repos") or {}).items():
        if not row.get("exists") or not row.get("commit"):
            problems.append("environment.lock.json missing external repo commit for %s" % name)
    if payload.get("build_artifacts", {}).get("dude_executable", {}).get("exists") is not True:
        problems.append("environment.lock.json missing DUDE executable")
    return problems


def _repos_lock_problems(run_root: Path) -> list[str]:
    candidates = [Path(run_root) / "repos.lock.json", Path("external_baselines/repos.lock.json")]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return ["missing repos.lock.json"]
    payload = _read_json(path)
    problems: list[str] = []
    for name, row in dict(payload.get("repos") or {}).items():
        if not row.get("exists") or not row.get("commit"):
            problems.append("repos.lock.json missing commit for %s" % name)
    expected = {"ACTIVE_ROOM_SEG_ROOT", "INCREMENTAL_DUDE_ROS_ROOT", "ROSE2_ROOT", "IPA_COVERAGE_ROOT"}
    missing = sorted(expected - set(dict(payload.get("repos") or {}).keys()))
    for name in missing:
        problems.append("repos.lock.json missing %s" % name)
    return problems


def _baseline_manifest_problems(run_root: Path) -> list[str]:
    path = Path(run_root) / "baseline_replay_manifest.json"
    if not path.exists():
        return ["missing baseline_replay_manifest.json"]
    payload = _read_json(path)
    problems: list[str] = []
    if int(payload.get("schema_version", -1)) < 2:
        problems.append("baseline_replay_manifest.json schema_version < 2")
    if payload.get("strict_native") is not True:
        problems.append("baseline_replay_manifest.json strict_native is not true")
    if payload.get("fallback_python") is True:
        problems.append("baseline_replay_manifest.json fallback_python is true")
    if payload.get("allow_baseline_failure") is True:
        problems.append("baseline_replay_manifest.json allow_baseline_failure is true")
    return problems


def _baseline_metadata_strings(run_root: Path) -> list[str]:
    import numpy as np

    rows: list[str] = []
    for path in Path(run_root).glob("*/baselines/*/roomseg_snapshots/*.npz"):
        try:
            with np.load(path, allow_pickle=False) as data:
                if "baseline_metadata_json" in data.files:
                    rows.append(str(data["baseline_metadata_json"]).lower())
        except Exception:
            continue
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate whether a comparison run is ready for paper metrics.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--require-csr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-native", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-original-commits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-no-placeholder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-environment-lock", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-repos-lock", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-strict-replay-manifest", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    report = validate_paper_ready(
        run_root=Path(args.run_root),
        eval_root=Path(args.eval_root),
        methods=[str(v) for v in args.methods],
        require_csr=bool(args.require_csr),
        require_native=bool(args.require_native),
        require_original_commits=bool(args.require_original_commits),
        require_no_placeholder=bool(args.require_no_placeholder),
        require_environment_lock=bool(args.require_environment_lock),
        require_repos_lock=bool(args.require_repos_lock),
        require_strict_replay_manifest=bool(args.require_strict_replay_manifest),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["paper_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
