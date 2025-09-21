"""Visualization subpackage.

Phases 0–2 minimal implementation:
- FrameData & AgentFrame dataclasses (pure snapshot)
- BaseRenderer interface
- ASCII renderer (always available)
- Optional pygame renderer (only if pygame installed and --gui flag used)

Higher phases (analytical overlays, recording) intentionally deferred.
"""

from .frame_data import FrameData, AgentFrame, build_frame  # noqa: F401
from .ascii_renderer import ASCIIRenderer  # noqa: F401

try:  # pragma: no cover - optional dependency
    from .pygame_renderer import PygameRenderer  # noqa: F401
except Exception:  # pragma: no cover
    PygameRenderer = None  # type: ignore
