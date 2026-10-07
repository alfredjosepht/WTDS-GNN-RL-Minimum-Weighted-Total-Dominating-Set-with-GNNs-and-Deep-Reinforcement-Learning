"""Common return type for every solver. `is_valid` is always computed by the checker."""
from __future__ import annotations

from dataclasses import dataclass, field

from .checker import assert_tds, verify
from .graph import Graph


@dataclass
class SolveResult:
    vertices: list
    size: int
    weight: float
    runtime: float
    is_valid: bool
    method: str
    info: dict = field(default_factory=dict)

    @classmethod
    def build(cls, graph: Graph, S, runtime: float, method: str, strict: bool = True, **info):
        S = sorted(int(v) for v in set(S))
        valid = verify(graph, S)
        if strict and not valid:
            assert_tds(graph, S, context=method)
        return cls(vertices=S, size=len(S), weight=float(graph.w[S].sum()) if S else 0.0,
                   runtime=float(runtime), is_valid=valid, method=method, info=info)

    def to_dict(self) -> dict:
        return dict(vertices=self.vertices, size=self.size, weight=self.weight,
                    runtime=self.runtime, is_valid=self.is_valid, method=self.method,
                    info={k: v for k, v in self.info.items() if _jsonable(v)})


def _jsonable(v) -> bool:
    return isinstance(v, (int, float, str, bool, type(None), list, dict))
