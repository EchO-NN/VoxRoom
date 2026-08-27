from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

GridCell = Tuple[int, int]


@dataclass
class PlannedPath:
    cells: List[GridCell]
    length_m: float

