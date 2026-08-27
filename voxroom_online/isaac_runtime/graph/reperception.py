from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Sequence

import numpy as np

from voxroom_online.isaac_runtime.graph.hcot_scorer import SubgraphScore
from voxroom_online.isaac_runtime.graph.paper_scene_graph import ObjectNode


class ReperceptionDecision(str, Enum):
    ACCEPT_GOAL = "ACCEPT_GOAL"
    REJECT_GOAL = "REJECT_GOAL"
    CONTINUE_OBSERVING = "CONTINUE_OBSERVING"


@dataclass
class ReperceptionState:
    candidate_id: str
    accumulated_credibility: float = 0.0
    num_reperception_steps: int = 0
    history: list = field(default_factory=list)


@dataclass
class ReperceptionResult:
    candidate_id: str
    detector_confidence: float
    s_k: float
    accumulated_credibility: float
    num_reperception_steps: int
    decision: ReperceptionDecision
    supporting_subgraphs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "detector_confidence": float(self.detector_confidence),
            "s_k": float(self.s_k),
            "last_s_k": float(self.s_k),
            "accumulated_credibility": float(self.accumulated_credibility),
            "num_reperception_steps": int(self.num_reperception_steps),
            "decision": self.decision.value,
            "supporting_subgraphs": list(self.supporting_subgraphs),
        }


class GraphReperceptionManager:
    def __init__(self, n_max: int = 10, s_thres: float = 0.8):
        self.n_max = max(1, int(n_max))
        self.s_thres = float(s_thres)
        self.states: Dict[str, ReperceptionState] = {}

    def update(
        self,
        candidate: ObjectNode,
        detector_confidence: float,
        subgraph_scores: Sequence[SubgraphScore],
    ) -> ReperceptionResult:
        s_k = compute_goal_candidate_credibility(candidate, detector_confidence, subgraph_scores)
        state = self.states.setdefault(candidate.id, ReperceptionState(candidate_id=candidate.id))
        state.accumulated_credibility += float(s_k)
        state.num_reperception_steps += 1
        if state.accumulated_credibility >= self.s_thres:
            decision = ReperceptionDecision.ACCEPT_GOAL
        elif state.num_reperception_steps >= self.n_max and state.accumulated_credibility < self.s_thres:
            decision = ReperceptionDecision.REJECT_GOAL
        else:
            decision = ReperceptionDecision.CONTINUE_OBSERVING
        result = ReperceptionResult(
            candidate_id=candidate.id,
            detector_confidence=float(detector_confidence),
            s_k=float(s_k),
            accumulated_credibility=float(state.accumulated_credibility),
            num_reperception_steps=int(state.num_reperception_steps),
            decision=decision,
            supporting_subgraphs=_supporting_subgraph_payload(subgraph_scores),
        )
        state.history.append(result.to_dict())
        return result


def compute_goal_candidate_credibility(
    candidate: ObjectNode,
    detector_confidence: float,
    subgraph_scores: Sequence[SubgraphScore],
    eps: float = 0.25,
) -> float:
    candidate_center = np.asarray(candidate.center_world[:2], dtype=np.float32)
    support = 0.0
    for score in subgraph_scores:
        sub_center = np.asarray(score.central_world[:2], dtype=np.float32)
        dist = max(float(np.linalg.norm(candidate_center - sub_center)), float(eps))
        support += float(score.p_sub) / dist
    return float(detector_confidence) * float(support)


def _supporting_subgraph_payload(subgraph_scores: Sequence[SubgraphScore]) -> list[dict]:
    rows = []
    for score in subgraph_scores:
        rows.append(
            {
                "subgraph_id": str(score.subgraph_id),
                "central_object_id": str(score.central_object_id),
                "p_sub": float(score.p_sub),
                "estimated_distance_m": float(score.estimated_distance_m),
            }
        )
    rows.sort(key=lambda item: float(item["p_sub"]), reverse=True)
    return rows[:8]
