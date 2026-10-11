"""The vision model's location line (nav/location.py): parsing Mapbox's reverse geocoding, background lookups only
after moving, retries after failures, and nothing stale."""

from __future__ import annotations

import threading

from nav.geo import LatLon, destination
from nav.gps import GpsFix
from nav.location import LocationContext, place_name

BAYSIDE: LatLon = (40.761171, -73.763707)

REPLY = {
    "features": [
        {
            "properties": {
                "feature_type": "neighborhood",
                "name": "Bayside",
                "context": {
                    "postcode": {"name": "11361"},
                    "neighborhood": {"name": "Bayside"},
                    "locality": {"name": "Queens"},
                    "place": {"name": "New York City"},
                    "district": {"name": "Queens County"},
                    "region": {"name": "New York"},
                    "country": {"name": "United States"},
                },
            }
        }
    ]
}


def test_place_name_from_a_reverse_geocoding_reply() -> None:
    assert place_name(REPLY) == "Bayside, Queens, New York City, New York"
    same_names = {
        "features": [{"properties": {"context": {"place": {"name": "Seattle"}, "region": {"name": "Seattle"}}}}]
    }
    assert place_name(same_names) == "Seattle"  # (no repeats)
    assert place_name({"features": []}) is None
    assert place_name({"features": [{"properties": {}}]}) is None
    assert place_name({}) is None


class Fake:
    def __init__(self, fail: int = 0) -> None:
        self.calls: list[LatLon] = []
        self.fail = fail
        self.release = threading.Event()
        self.release.set()

    def __call__(self, at: LatLon) -> str | None:
        self.release.wait(2.0)
        self.calls.append(at)
        if self.fail > 0:
            self.fail -= 1
            raise RuntimeError("offline")
        return f"place {len(self.calls)}"


def fix_at(here: LatLon) -> GpsFix:
    return GpsFix(0.0, here[0], here[1], 10.0, None)


def test_looked_up_in_the_background_once_and_again_after_moving() -> None:
    reverser = Fake()
    here = [BAYSIDE]
    location = LocationContext(reverser, lambda: fix_at(here[0]), refresh_m=400.0)
    assert location.context() == []  # nothing known yet: no line, no waiting
    location.update()
    location.wait_idle()
    assert location.context() == ["Location (from GPS): place 1."]
    here[0] = destination(BAYSIDE, 90.0, 300.0)
    location.update()
    location.wait_idle()
    assert len(reverser.calls) == 1 and location.context() == ["Location (from GPS): place 1."]  # 300 m: same place
    here[0] = destination(BAYSIDE, 90.0, 500.0)
    location.update()
    location.wait_idle()
    assert len(reverser.calls) == 2 and location.context() == ["Location (from GPS): place 2."]


def test_a_question_never_waits_for_a_lookup_and_one_runs_at_a_time() -> None:
    reverser = Fake()
    reverser.release.clear()  # the lookup hangs
    location = LocationContext(reverser, lambda: fix_at(BAYSIDE))
    location.update()
    location.update()
    assert location.context() == []  # answered at once, without a place
    reverser.release.set()
    location.wait_idle()
    assert len(reverser.calls) == 1 and location.context() == ["Location (from GPS): place 1."]


def test_a_failed_lookup_is_retried_later_and_a_stale_place_is_dropped() -> None:
    now = [0.0]
    reverser = Fake(fail=1)
    here = [BAYSIDE]
    gps: list[GpsFix | None] = [fix_at(BAYSIDE)]
    location = LocationContext(reverser, lambda: gps[0], refresh_m=400.0, retry_s=60.0, clock=lambda: now[0])
    location.update()
    location.wait_idle()
    assert location.context() == []  # failed
    location.update()
    location.wait_idle()
    assert len(reverser.calls) == 1  # not again before retry_s
    now[0] = 61.0
    location.update()
    location.wait_idle()
    assert location.context() == ["Location (from GPS): place 2."]
    here[0] = destination(BAYSIDE, 0.0, 1000.0)  # far from where it was looked up, before the next lookup lands
    gps[0] = fix_at(here[0])
    assert location.context() == []
    gps[0] = None  # no GPS: no line
    assert location.context() == []
