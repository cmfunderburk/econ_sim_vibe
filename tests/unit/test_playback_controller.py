import time
import math
from typing import Optional, List

from src.visualization.playback import PlaybackController, FrameStream
from src.visualization.frame_data import FrameData


class DummyStream(FrameStream):  # type: ignore
    def __init__(self, max_frames: int = 5):
        self.frames: List[FrameData] = []
        for i in range(max_frames):
            self.frames.append(
                FrameData(
                    round=i + 1,
                    grid_width=1,
                    grid_height=1,
                    market_x0=0,
                    market_y0=0,
                    market_width=1,
                    market_height=1,
                    agents=[],
                    prices=[],
                    participation_count=0,
                    total_agents=0,
                )
            )
        self.last_round: int = 0  # matches LogReplayStream semantics
        self._max_round: int = len(self.frames)

    def next_frame(self) -> Optional[FrameData]:
        if self.last_round >= len(self.frames):
            return None
        self.last_round += 1
        return self.frames[self.last_round - 1]

    def prev_frame(self) -> Optional[FrameData]:  # pragma: no cover - simple rewind
        if not self.frames:
            return None
        if self.last_round <= 1:
            self.last_round = 1
            return self.frames[0]
        self.last_round -= 1
        return self.frames[self.last_round - 1]

    def seek(self, round_number: int) -> None:  # pragma: no cover
        if not self.frames:
            self.last_round = 0
            return
        clamped = max(0, min(round_number, len(self.frames)))
        self.last_round = clamped

    def reset(self) -> None:  # pragma: no cover
        self.last_round = 0


def test_play_pause_behavior(monkeypatch):
    stream = DummyStream(max_frames=3)
    controller = PlaybackController(stream=stream, rounds_per_second=30.0)

    # Force immediate frame production by manipulating last_tick
    controller.last_tick -= controller.frame_interval
    f1 = controller.update()
    assert f1 is not None and f1.round == 1

    controller.toggle_play()  # pause
    assert controller.is_playing is False

    # While paused without step, no new frame even if interval passed
    controller.last_tick -= controller.frame_interval * 2
    assert controller.update() is None

    # Single step while paused advances exactly one
    f2 = controller.step_once()
    assert f2 is not None and f2.round == 2

    # Further update without another step returns None
    assert controller.update() is None

    # Resume play advances next frame after interval
    controller.toggle_play()
    controller.last_tick -= controller.frame_interval
    f3 = controller.update()
    assert f3 is not None and f3.round == 3


def test_step_back_and_jump():
    stream = DummyStream(max_frames=5)
    controller = PlaybackController(stream=stream, rounds_per_second=5.0)

    controller.last_tick -= controller.frame_interval
    assert controller.update() and controller.current_round == 1
    controller.last_tick -= controller.frame_interval
    assert controller.update() and controller.current_round == 2

    # Step backward returns previous frame and pauses
    back_frame = controller.step_back()
    assert back_frame is not None and back_frame.round == 1
    assert controller.current_round == 1
    assert controller.is_playing is False

    # Jump forward by 3 rounds
    jump_frame = controller.jump(3)
    assert jump_frame is not None and jump_frame.round == 4
    assert controller.current_round == 4

    # Jump past end clamps to last frame
    end_frame = controller.goto(99)
    assert end_frame is not None and end_frame.round == 5
    assert controller.current_round == 5


def test_speed_clamping():
    stream = DummyStream(max_frames=1)
    controller = PlaybackController(stream=stream, rounds_per_second=1000.0)
    # Should clamp to <= 60
    assert controller.rounds_per_second <= 60.0

    controller.set_speed(0.01)
    assert controller.rounds_per_second >= 0.25

    controller.set_speed(10.0)
    assert math.isclose(controller.rounds_per_second, 10.0)


def test_no_advance_before_interval():
    stream = DummyStream(max_frames=2)
    controller = PlaybackController(
        stream=stream, rounds_per_second=5.0
    )  # 0.2s interval
    # Do not adjust last_tick; elapsed < frame_interval
    assert controller.update() is None
    # Force enough time to pass
    controller.last_tick -= controller.frame_interval
    frame = controller.update()
    assert frame is not None and frame.round == 1
