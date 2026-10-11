"""Where we are, for the vision model: "Location: Bayside, Queens, New York City, New York."

Rules depend on the place (no right turn on red anywhere in New York City unless a sign allows it; curb colors in
California; school-zone hours), and the model can't tell the city from a frame. `LocationContext` turns the GPS
fix into that one line by reverse geocoding (Mapbox Geocoding v6), in a background thread, so a question never
waits for it: `update()` (called often, e.g. once a second) starts a lookup when we have moved `refresh_m` since
the last one, and `context()` returns the latest line while its fix is near enough to where we are.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from core.logging_setup import get_logger
from nav.geo import LatLon, haversine_m
from nav.gps import GpsFix

log = get_logger(__name__)

REVERSE_URL = "https://api.mapbox.com/search/geocode/v6/reverse"
PLACE_LEVELS = ("neighborhood", "locality", "place", "region")  # most local first


def place_name(data: dict[str, Any]) -> str | None:
    """'Bayside, Queens, New York City, New York' from a Geocoding v6 reverse reply (None if it has none)."""
    features = data.get("features")
    if not isinstance(features, list) or not features:
        return None
    properties = features[0].get("properties") if isinstance(features[0], dict) else None
    context = properties.get("context") if isinstance(properties, dict) else None
    if not isinstance(context, dict):
        return None
    names: list[str] = []
    for level in PLACE_LEVELS:
        entry = context.get(level)
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, str) and name.strip() and name not in names:
            names.append(name.strip())
    return ", ".join(names) or None


class Reverser(Protocol):
    """Reverse geocoding: the place name at a position (None: nothing known there)."""

    def __call__(self, at: LatLon) -> str | None: ...


def mapbox_reverser(token: str, timeout_s: float = 5.0) -> Reverser:
    """A `Reverser` on Mapbox Geocoding v6 (raises on network or HTTP errors)."""
    import httpx

    client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=3.0))

    def reverse(at: LatLon) -> str | None:
        params = {
            "latitude": f"{at[0]:.6f}",
            "longitude": f"{at[1]:.6f}",
            "types": ",".join(PLACE_LEVELS),
            "language": "en",
            "access_token": token,
        }
        response = client.get(REVERSE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return place_name(data) if isinstance(data, dict) else None

    return reverse


class LocationContext:
    """The vision model's location line, kept up to date from the GPS in the background."""

    def __init__(
        self,
        reverser: Reverser,
        gps: Callable[[], GpsFix | None],
        refresh_m: float = 400.0,
        retry_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """`refresh_m`: look up again after moving this far (also the furthest the line is used from where it was
        looked up); `retry_s`: wait this long after a failed lookup."""
        self._reverser = reverser
        self._gps = gps
        self.refresh_m = refresh_m
        self.retry_s = retry_s
        self._clock = clock
        self._lock = threading.Lock()
        self._place: tuple[LatLon, str] | None = None  # where we looked up, and the name
        self._busy = False
        self._failed_at: float | None = None

    def update(self) -> None:
        """Start a lookup if we have moved far enough from the last one (never blocks)."""
        fix = self._gps()
        if fix is None:
            return
        here = fix.latlon
        with self._lock:
            if self._busy:
                return
            if self._failed_at is not None and self._clock() - self._failed_at < self.retry_s:
                return
            if self._place is not None and haversine_m(self._place[0], here) < self.refresh_m:
                return
            self._busy = True
        threading.Thread(target=self._lookup, args=(here,), name="location", daemon=True).start()

    def _lookup(self, here: LatLon) -> None:
        try:
            name = self._reverser(here)
        except Exception as exc:  # noqa: BLE001 - offline, rate limited...: the answers just go without it
            log.warning("location lookup failed; retrying later", error=f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._busy, self._failed_at = False, self._clock()
            return
        with self._lock:
            self._busy, self._failed_at = False, None
            if name:
                self._place = (here, name)
        if name:
            log.info("location", place=name)

    def wait_idle(self, timeout_s: float = 5.0) -> None:
        """Wait for a lookup in flight (tests and lock-step replays)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._busy:
                    return
            time.sleep(0.01)

    def context(self) -> list[str]:
        """["Location (from GPS): <place>."] when we know where we are, else []."""
        fix = self._gps()
        with self._lock:
            place = self._place
        if fix is None or place is None or haversine_m(place[0], fix.latlon) > 2 * self.refresh_m:
            return []
        return [f"Location (from GPS): {place[1]}."]
