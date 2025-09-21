from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
from .frame_data import FrameData


class Renderer(Protocol):  # Structural protocol for typing convenience
    def render(self, frame: FrameData) -> None: ...  # pragma: no cover - interface


class BaseRenderer(ABC):
    """Abstract base renderer with a uniform render() method."""

    @abstractmethod
    def render(self, frame: FrameData) -> None:  # pragma: no cover - interface
        raise NotImplementedError
