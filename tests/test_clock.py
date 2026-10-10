"""MediaClock and WallClock behaviour with a fake monotonic time source."""

from __future__ import annotations

import threading

import pytest

from core.clock import MediaClock, WallClock


class FakeTime:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_wall_clock_counts_from_creation() -> None:
    ft = FakeTime()
    clock = WallClock(ft)
    assert clock.now() == 0.0
    ft.t += 2.5
    assert clock.now() == pytest.approx(2.5)


def test_media_clock_starts_paused_then_plays() -> None:
    ft = FakeTime()
    clock = MediaClock(start=10.0, monotonic=ft)
    assert clock.paused
    ft.t += 5
    assert clock.now() == 10.0  # paused: no advance
    clock.play()
    ft.t += 1.5
    assert clock.now() == pytest.approx(11.5)


def test_media_clock_pause_resume_and_seek() -> None:
    ft = FakeTime()
    clock = MediaClock(monotonic=ft)
    clock.play()
    ft.t += 3
    clock.pause()
    ft.t += 100
    assert clock.now() == pytest.approx(3.0)
    clock.seek(20.0)
    assert clock.now() == pytest.approx(20.0)  # still paused after seek
    clock.play()
    ft.t += 2
    assert clock.now() == pytest.approx(22.0)
    clock.seek(-5)  # clamped
    assert clock.now() == 0.0
    ft.t += 1
    assert clock.now() == pytest.approx(1.0)  # seek keeps playing state


def test_media_clock_rate() -> None:
    ft = FakeTime()
    clock = MediaClock(rate=4.0, monotonic=ft)
    clock.play()
    ft.t += 2
    assert clock.now() == pytest.approx(8.0)
    with pytest.raises(ValueError):
        MediaClock(rate=0)


def test_media_clock_is_monotonic_under_concurrent_readers() -> None:
    clock = MediaClock()
    clock.play()
    errors: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        last = -1.0
        while not stop.is_set():
            now = clock.now()
            if now < last:
                errors.append(f"went backwards {last} -> {now}")
            last = now

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(200):  # pause/resume from another thread must never make time jump backwards
        clock.pause()
        clock.play()
    stop.set()
    for t in threads:
        t.join(2)
    assert not errors


def test_play_and_pause_are_idempotent() -> None:
    ft = FakeTime()
    clock = MediaClock(monotonic=ft)
    clock.play()
    ft.t += 1
    clock.play()  # must not re-anchor and lose the elapsed second
    assert clock.now() == pytest.approx(1.0)
    clock.pause()
    clock.pause()
    assert clock.now() == pytest.approx(1.0)
