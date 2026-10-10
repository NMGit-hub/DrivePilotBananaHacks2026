"""Mapbox geocoding (v6) and directions (v5, driving-traffic) client, plus a fixture-backed stand-in.

`MapboxClient.route(origin, place_text)` turns "Redmond Town Center" into a `Route` from where we are: geocode
(biased toward our position), then directions with steps, banner instructions and lane data. Our heading is
passed as the origin bearing so the route doesn't start with a U-turn. Failures raise `NavError` with a
message fit to log; the handler speaks a short version. The token never appears in logs (third-party loggers
are kept at WARNING and every log line is redacted).

`FixtureClient` answers from saved responses with the same interface (tests, offline replays).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from core.logging_setup import get_logger
from nav.geo import LatLon, haversine_m
from nav.route import Route, RouteError, parse_directions

log = get_logger(__name__)

GEOCODE_URL = "https://api.mapbox.com/search/geocode/v6/forward"
SEARCHBOX_URL = "https://api.mapbox.com/search/searchbox/v1/forward"
DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox/{profile}/{coordinates}"
_STOPWORDS = {"the", "a", "an", "to", "of", "at", "in"}


class NavError(RuntimeError):
    """Geocoding or routing failed."""


@dataclass(frozen=True, slots=True)
class Place:
    """A geocoded destination."""

    name: str  # short, e.g. "Redmond Town Center"
    address: str  # full address for logs/screen
    location: LatLon
    kind: str = ""  # "poi", "address", "street", "neighborhood", ...


def _places(data: dict[str, Any]) -> list[Place]:
    out = []
    for feature in data.get("features") or []:
        props = feature.get("properties") or {}
        coordinates = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coordinates) < 2 or not props.get("name"):
            continue
        name = str(props["name"])
        address = str(props.get("full_address") or props.get("place_formatted") or name)
        out.append(
            Place(name, address, (float(coordinates[1]), float(coordinates[0])), str(props.get("feature_type", "")))
        )
    return out


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower().replace("'s", "")) if w not in _STOPWORDS}


def pick_place(query: str, candidates: list[Place], near: LatLon | None) -> Place:
    """The candidate whose name best matches the spoken words, weighed against distance from us."""
    spoken = _tokens(query)

    def score(place: Place) -> float:
        name = _tokens(place.name)
        overlap = len(spoken & name) / max(len(spoken), 1)  # share of the spoken words in the name
        extra = len(name - spoken) / max(len(name), 1)  # words in the name nobody said
        value = overlap - 0.2 * extra + (0.5 if name == spoken else 0.0)
        value += 0.1 if place.kind in ("poi", "address") else 0.0
        if near is not None:
            km = haversine_m(near, place.location) / 1000.0
            value -= 0.3 * math.log10(1.0 + km / 5.0) + (1.0 if km > 300.0 else 0.0)
        return value

    best = max(candidates, key=score)
    log.info(
        "destination chosen",
        query=query,
        name=best.name,
        address=best.address,
        kind=best.kind,
        candidates=len(candidates),
    )
    return best


class RoutingClient(Protocol):
    """What the navigation handler needs from a routing service."""

    def geocode(self, query: str, near: LatLon | None = None) -> Place:
        """The best match for a spoken place name."""

    def route(self, origin: LatLon, destination: Place, heading: float | None = None) -> Route:
        """A driving route from `origin` to the place."""


class MapboxClient:
    """Mapbox Search (geocoding v6) and Directions (v5) over HTTP."""

    def __init__(
        self,
        token: str,
        profile: str = "driving-traffic",
        timeout_s: float = 5.0,
        country: str | None = "us",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._token = token
        self.profile = profile
        self.country = country
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=3.0), transport=transport)

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.get(url, params={**params, "access_token": self._token})
        except httpx.HTTPError as exc:
            raise NavError(f"mapbox: {type(exc).__name__}") from exc
        if response.status_code == 401:
            raise NavError("mapbox: the access token was rejected (check MAPBOX_ACCESS_TOKEN)")
        if response.status_code != 200:
            raise NavError(f"mapbox: HTTP {response.status_code}: {response.text[:200]}")
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise NavError("mapbox: invalid JSON") from exc
        if not isinstance(data, dict):
            raise NavError("mapbox: unexpected reply")
        return data

    def geocode(self, query: str, near: LatLon | None = None) -> Place:
        """The best match for a spoken place name near us.

        Businesses and landmarks ("Costco", "Swedish Issaquah") are only in the Search Box API, streets and
        addresses are better in Geocoding v6, and either one's single best guess can be absurd ("Space Needle" →
        a shop in Arizona). So both are asked for a few candidates, and the one whose name best matches the words
        spoken, weighed against distance from us, wins (`pick_place`)."""
        params: dict[str, Any] = {"q": query, "language": "en"}
        if near is not None:
            params["proximity"] = f"{near[1]:.6f},{near[0]:.6f}"
        if self.country:
            params["country"] = self.country
        candidates: list[Place] = []
        errors: list[str] = []
        for url, limit in ((SEARCHBOX_URL, 5), (GEOCODE_URL, 3)):
            try:
                candidates += _places(self._get(url, {**params, "limit": limit}))
            except NavError as exc:
                errors.append(str(exc))
        if not candidates:
            raise NavError(errors[0] if errors else f"no place found for {query!r}")
        return pick_place(query, candidates, near)

    def directions(self, origin: LatLon, destination: LatLon, heading: float | None = None) -> dict[str, Any]:
        """The raw directions response (kept for fixtures)."""
        coordinates = f"{origin[1]:.6f},{origin[0]:.6f};{destination[1]:.6f},{destination[0]:.6f}"
        params: dict[str, Any] = {
            "steps": "true",
            "banner_instructions": "true",
            "geometries": "geojson",
            "overview": "full",
            "alternatives": "false",
            "language": "en",
            "voice_units": "imperial",
            "roundabout_exits": "true",  # the ring is its own step, ending where we leave it (exit counting)
        }
        if heading is not None:
            params["bearings"] = f"{round(heading) % 360},60;"
        return self._get(DIRECTIONS_URL.format(profile=self.profile, coordinates=coordinates), params)

    def route(self, origin: LatLon, destination: Place, heading: float | None = None) -> Route:
        """A route from `origin` to `destination`."""
        data = self.directions(origin, destination.location, heading)
        try:
            return parse_directions(data, destination.name)
        except RouteError as exc:
            raise NavError(str(exc)) from exc

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


class FixtureClient:
    """Answers from saved responses: {"places": {query: {...}}, "routes": {name: directions-response}}."""

    def __init__(self, places: dict[str, Place], routes: dict[str, dict[str, Any]]) -> None:
        self.places = {k.lower(): v for k, v in places.items()}
        self.routes = routes
        self.requests: list[tuple[LatLon, str]] = []

    @classmethod
    def from_files(cls, directory: Path) -> FixtureClient:
        """Load every `<name>.json` directions response in a folder; the place is named after the file."""
        places, routes = {}, {}
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            response = data.get("directions", data)
            place = data.get("place")
            name = path.stem.replace("_", " ")
            if place:
                places[name] = Place(place["name"], place.get("address", place["name"]), tuple(place["location"]))
            routes[name] = response
        return cls(places, routes)

    def geocode(self, query: str, near: LatLon | None = None) -> Place:
        """A saved place whose name contains the query (or the reverse)."""
        q = query.lower().strip()
        for key, place in self.places.items():
            if q in key or key in q or q in place.name.lower():
                return place
        raise NavError(f"no place found for {query!r}")

    def route(self, origin: LatLon, destination: Place, heading: float | None = None) -> Route:
        """The saved route to that place."""
        self.requests.append((origin, destination.name))
        for key, response in self.routes.items():
            if key in destination.name.lower() or destination.name.lower() in key:
                return parse_directions(response, destination.name)
        raise NavError(f"no saved route to {destination.name!r}")
