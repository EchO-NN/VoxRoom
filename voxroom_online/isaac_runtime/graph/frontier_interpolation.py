from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from voxroom_online.isaac_runtime.graph.hcot_scorer import SubgraphScore
from voxroom_online.isaac_runtime.mapping.frontier import FrontierCluster


@dataclass
class FrontierScore:
    frontier_id: str
    frontier: FrontierCluster
    score: float
    top_supporting_subgraphs: List[dict] = field(default_factory=list)
    explanation: str = ""


def score_frontiers_by_subgraphs(
    frontiers: Sequence[FrontierCluster],
    subgraph_scores: Sequence[SubgraphScore],
    eps: float = 0.25,
    top_k: int = 3,
) -> List[FrontierScore]:
    results: List[FrontierScore] = []
    for idx, frontier in enumerate(frontiers):
        frontier_center = _frontier_center_world(frontier)
        contributions = []
        total = 0.0
        for score in subgraph_scores:
            sub_center = np.asarray(score.central_world[:2], dtype=np.float32)
            dist = max(float(np.linalg.norm(frontier_center - sub_center)), float(eps))
            contribution = float(score.p_sub) / dist
            total += contribution
            contributions.append(
                {
                    "subgraph_id": score.subgraph_id,
                    "central_object_id": score.central_object_id,
                    "p_sub": float(score.p_sub),
                    "estimated_distance_m": float(score.estimated_distance_m),
                    "distance_to_frontier_m": dist,
                    "contribution": contribution,
                    "summary_reason": score.summary_reason,
                }
            )
        contributions.sort(key=lambda item: float(item["contribution"]), reverse=True)
        top = contributions[: max(0, int(top_k))]
        explanation = "; ".join(str(item.get("summary_reason", "")) for item in top if item.get("summary_reason"))
        results.append(
            FrontierScore(
                frontier_id="frontier_%d" % idx,
                frontier=frontier,
                score=float(total),
                top_supporting_subgraphs=top,
                explanation=explanation,
            )
        )
    return results


def select_highest_score_frontier(frontier_scores: Sequence[FrontierScore]) -> Optional[FrontierScore]:
    if not frontier_scores:
        return None
    return max(frontier_scores, key=lambda item: float(item.score))


def frontier_debug_payload(frontier_scores: Sequence[FrontierScore]) -> dict:
    selected = select_highest_score_frontier(frontier_scores)
    return {
        "selected_frontier_id": selected.frontier_id if selected is not None else None,
        "frontier_scores": [
            {
                "frontier_id": item.frontier_id,
                "score": float(item.score),
                "center_world": list(_frontier_center_world(item.frontier)),
            }
            for item in frontier_scores
        ],
        "top_supporting_subgraphs": [] if selected is None else selected.top_supporting_subgraphs,
        "explanation": "" if selected is None else selected.explanation,
    }


def _frontier_center_world(frontier: FrontierCluster) -> np.ndarray:
    center = getattr(frontier, "center_world", (0.0, 0.0))
    return np.asarray([float(center[0]), float(center[1])], dtype=np.float32)
