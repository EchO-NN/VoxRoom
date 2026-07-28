def _edge_contract(edge):
    return (
        int(edge["stable_id"]),
        int(edge["source_stable_id"]),
        int(edge["target_stable_id"]),
        tuple(int(value) for value in edge["way_point"]),
    )


def crossing_evidence_survives(evidence, topology_snapshot):
    final_edges = {
        _edge_contract(edge)
        for edge in topology_snapshot.get("edges", [])
    }
    segments = evidence.get("segments", [])
    if not segments:
        return False
    for segment in segments:
        if not segment.get("confirmed"):
            return False
        contract = (
            int(segment["edge_stable_id"]),
            int(segment["source_node_stable_id"]),
            int(segment["target_node_stable_id"]),
            tuple(int(value) for value in segment["target_waypoint"]),
        )
        if contract not in final_edges:
            return False
    return True


def surviving_crossing_count(crossing_evidence, topology_snapshot):
    return sum(
        crossing_evidence_survives(evidence, topology_snapshot)
        for evidence in crossing_evidence
    )
