"""Generative methods that share one DiT, one data pipeline and one trainer."""

from tinydit.methods.base import GenerativeMethod, MethodKind
from tinydit.methods.ddpm import Ddpm
from tinydit.methods.flow_matching import FlowMatching, OdeSolver
from tinydit.methods.mean_flow import MeanFlow
from tinydit.methods.shortcut import Shortcut

_METHODS = {MethodKind.DDPM: Ddpm, MethodKind.FLOW_MATCHING: FlowMatching, MethodKind.MEANFLOW: MeanFlow,
            MethodKind.SHORTCUT: Shortcut}


def build_method(kind: MethodKind) -> GenerativeMethod:
    """The method with its default settings."""
    return _METHODS[kind]()


__all__ = ["Ddpm", "FlowMatching", "GenerativeMethod", "MeanFlow", "MethodKind", "OdeSolver", "Shortcut",
           "build_method"]
