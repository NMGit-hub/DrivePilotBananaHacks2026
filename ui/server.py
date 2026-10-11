"""The local web server: the screen, the phone GPS page and their small API (FastAPI + uvicorn).

    GET  /            the V2 screen (`ui.overlays`: video with overlays, ui/static/v2.html) or the V1 one
    GET  /v1, /v2     each screen explicitly. V1: plain video plus the question and answer as text (index.html)
    GET  /gps         the phone page that sends its location once a second (ui/static/gps.html)
    GET  /video.mjpg  the processed video as MJPEG (V1 screen)
    WS   /ws/video    the V2 screen's video: each frame with its overlay packet as one binary message
    WS   /ws          conversation history on connect, then every new message and status update
    POST /api/ask     {"text": ...}: a typed question (hotkey Q)
    POST /api/control {"action": "pause"} | {"action": "seek", "delta_s": ±5}: replay hotkeys
    POST /api/gps     {"lat", "lon", "speed", "heading", "accuracy"}: a phone fix
    POST /api/places  {"q": ...}: places for the destination box as the driver types
    POST /api/destination  {"query": ..., "place": {...} | null}: navigate there (a typed place or a suggestion)
    POST /api/route   {"action": "cancel"}: end navigation
    POST /api/wake    {"mode": "hey_drivepilot" | "hey" | "none"}: the wake mode
    POST /api/alerts  {"name": ..., "on": bool}: switch one of the alerts DrivePilot says on its own
    GET  /api/image/N a memory thumbnail an answer came from (JPEG)
    GET  /api/state   history + status (debugging, tests)

The server listens on `ui.host` (127.0.0.1 by default); the phone reaches it over USB with
`adb reverse tcp:8000 tcp:8000`, as `localhost`, where browsers allow location access. Other web pages the
browser has open can't use it: requests must name an allowed host (a DNS-rebinding page names its own), POSTs
and the WebSocket must come from the page's own origin (403 otherwise), and POST bodies must be JSON (415).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from core.logging_setup import get_logger
from ui.hub import CLOSE, UiHub

log = get_logger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
BOUNDARY = "frame"


class UiActions(Protocol):
    """What the page can ask the app to do."""

    def ask_typed(self, text: str) -> None:
        """Answer a typed question (must not block: answer on another thread)."""

    def toggle_pause(self) -> bool | None:
        """Pause/resume a replay; returns the new state (None for a live camera)."""

    def seek(self, delta_s: float) -> bool:
        """Skip a replay by `delta_s` seconds; False for a live camera."""

    def gps_fix(
        self, lat: float, lon: float, speed: float | None, heading: float | None, accuracy: float | None
    ) -> bool:
        """A fix from the phone; False if the app doesn't use phone GPS."""

    def set_wake_mode(self, mode: str) -> bool:
        """Switch the wake mode; False without voice input."""

    def set_alert(self, name: str, on: bool) -> None:
        """Switch one unrequested alert on or off (ValueError for an unknown name)."""

    def suggest_places(self, query: str) -> list[dict[str, Any]]:
        """Places matching what was typed (blocks on the network: called off the event loop)."""

    def set_destination(self, query: str, place: dict[str, Any] | None) -> bool:
        """Navigate to a typed place or a picked suggestion (must not block); False without navigation."""

    def cancel_route(self) -> bool:
        """End navigation; False without navigation."""


class AskBody(BaseModel):
    """A typed question."""

    text: str = Field(min_length=1, max_length=500)


class ControlBody(BaseModel):
    """A replay control."""

    action: Literal["pause", "seek"]
    delta_s: float = Field(0.0, ge=-600.0, le=600.0)


class GpsBody(BaseModel):
    """One position from the phone's browser (Geolocation API units)."""

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    speed: float | None = Field(None, ge=0.0, le=120.0, description="m/s")
    heading: float | None = Field(None, ge=0.0, le=360.0, description="degrees clockwise from north")
    accuracy: float | None = Field(None, ge=0.0, description="m")


class PlacesBody(BaseModel):
    """What the driver has typed into the destination box."""

    q: str = Field(max_length=200)


class PlaceBody(BaseModel):
    """A suggestion the driver picked."""

    name: str = Field(min_length=1, max_length=200)
    address: str = Field("", max_length=400)
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    kind: str = Field("", max_length=40)


class DestinationBody(BaseModel):
    """Navigate here."""

    query: str = Field(min_length=1, max_length=200)
    place: PlaceBody | None = None


class RouteBody(BaseModel):
    """A navigation control."""

    action: Literal["cancel"]


class WakeBody(BaseModel):
    """The wake mode."""

    mode: Literal["hey_drivepilot", "hey", "none"]


class AlertSwitchBody(BaseModel):
    """One unrequested alert on or off."""

    name: str = Field(min_length=1, max_length=40)
    on: bool


LOCAL_HOSTS = ("localhost", "127.0.0.1")


def allowed_hosts(host: str) -> tuple[str, ...]:
    """Host names the page may be reached by: the local ones, plus the configured listening address."""
    extra = () if host in ("0.0.0.0", "") else (host,)
    return tuple(dict.fromkeys((*LOCAL_HOSTS, *extra)))


def _origin_ok(origin: str | None, hosts: Sequence[str]) -> bool:
    """A browser request from our own page (or a non-browser client, which sends no Origin)."""
    return origin is None or urlparse(origin).hostname in hosts


async def require_json(request: Request) -> None:
    """Reject non-JSON bodies (a cross-origin form post can't set this content type without CORS)."""
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="send JSON")


def create_server(hub: UiHub, actions: UiActions, hosts: Sequence[str] = LOCAL_HOSTS, overlays: bool = True) -> FastAPI:
    """The FastAPI app for the page, answering only to `hosts`; `/` is the V2 overlay page (`overlays`) or the
    plain V1 page (both stay reachable at /v2 and /v1)."""

    async def same_origin(request: Request) -> None:
        if not _origin_ok(request.headers.get("origin"), hosts):
            raise HTTPException(status_code=403, detail="not from this page")

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        hub.bind_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(title="DrivePilot", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(hosts))
    guarded = [Depends(same_origin), Depends(require_json)]

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / ("v2.html" if overlays else "index.html"), headers={"Cache-Control": "no-store"})

    @app.get("/v1")
    async def page_v1() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/v2")
    async def page_v2() -> FileResponse:
        return FileResponse(STATIC / "v2.html", headers={"Cache-Control": "no-store"})

    @app.get("/gps")
    async def gps_page() -> FileResponse:
        return FileResponse(STATIC / "gps.html", headers={"Cache-Control": "no-store"})

    @app.get("/video.mjpg")
    async def video(request: Request) -> StreamingResponse:
        async def frames() -> AsyncIterator[bytes]:
            seen = -1
            while not hub.stopped and not await request.is_disconnected():
                seq, jpeg = hub.latest_jpeg()
                if jpeg is None or seq == seen:
                    await asyncio.sleep(0.02)
                    continue
                seen = seq
                yield (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/ws/video")
    async def ws_video(socket: WebSocket) -> None:
        """The V2 page's video: each new frame with its overlay as one binary message (ui/hub.py)."""
        if not _origin_ok(socket.headers.get("origin"), hosts):
            await socket.close(code=1008)
            return
        await socket.accept()
        closed = asyncio.create_task(_until_disconnected(socket))
        seen = -1
        try:
            while not hub.stopped and not closed.done():
                seq, message = hub.latest_message()
                if message is None or seq == seen:
                    await asyncio.sleep(0.01)
                    continue
                seen = seq
                await socket.send_bytes(message)  # (waits for a slow page: it then skips to the newest frame)
            if not closed.done():
                await socket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            closed.cancel()

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if not _origin_ok(socket.headers.get("origin"), hosts):  # WebSockets aren't covered by CORS
            await socket.close(code=1008)
            return
        await socket.accept()
        queue = hub.subscribe()
        closed = asyncio.create_task(_until_disconnected(socket))  # notices a closed page even when idle
        try:
            await socket.send_json(
                {"type": "hello", "session": hub.session, "history": hub.history(), "status": hub.status()}
            )
            while True:
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({getter, closed}, return_when=asyncio.FIRST_COMPLETED)
                if closed in done:
                    getter.cancel()
                    break
                message = getter.result()
                if message.get("type") == CLOSE:  # the app is shutting down
                    await socket.close()
                    break
                await socket.send_json(message)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            closed.cancel()
            hub.unsubscribe(queue)

    @app.post("/api/ask", dependencies=guarded)
    async def ask(body: AskBody) -> JSONResponse:
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="empty question")
        actions.ask_typed(text)
        return JSONResponse({"ok": True})

    @app.post("/api/control", dependencies=guarded)
    async def control(body: ControlBody) -> JSONResponse:
        if body.action == "pause":
            paused = actions.toggle_pause()
            return JSONResponse({"ok": paused is not None, "paused": paused})
        return JSONResponse({"ok": actions.seek(body.delta_s)})

    @app.post("/api/gps", dependencies=guarded)
    async def gps(body: GpsBody) -> JSONResponse:
        used = actions.gps_fix(body.lat, body.lon, body.speed, body.heading, body.accuracy)
        return JSONResponse({"ok": used})

    @app.post("/api/places", dependencies=guarded)
    async def places(body: PlacesBody) -> JSONResponse:
        query = body.q.strip()
        found = await asyncio.to_thread(actions.suggest_places, query) if len(query) >= 2 else []
        return JSONResponse({"places": found})

    @app.post("/api/destination", dependencies=guarded)
    async def destination(body: DestinationBody) -> JSONResponse:
        query = body.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="empty destination")
        place = body.place.model_dump() if body.place is not None else None
        return JSONResponse({"ok": actions.set_destination(query, place)})

    @app.post("/api/route", dependencies=guarded)
    async def route(body: RouteBody) -> JSONResponse:
        return JSONResponse({"ok": actions.cancel_route()})

    @app.post("/api/wake", dependencies=guarded)
    async def wake(body: WakeBody) -> JSONResponse:
        return JSONResponse({"ok": actions.set_wake_mode(body.mode)})

    @app.post("/api/alerts", dependencies=guarded)
    async def alerts(body: AlertSwitchBody) -> JSONResponse:
        try:
            actions.set_alert(body.name, body.on)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse({"ok": True})

    @app.get("/api/image/{image_id}")
    async def image(image_id: int) -> Response:
        jpeg = hub.image(image_id)
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no such image")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse({"history": hub.history(), "status": hub.status()})

    return app


async def _until_disconnected(socket: WebSocket) -> None:
    """Read (and ignore) what the page sends until it goes away."""
    while True:
        message = await socket.receive()
        if message["type"] == "websocket.disconnect":
            return


class UiServer:
    """Runs uvicorn on a background thread."""

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        import uvicorn

        self.host, self.port = host, port
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="on",
            timeout_graceful_shutdown=1,  # open video streams never finish by themselves
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="ui-server", daemon=True)

    @property
    def url(self) -> str:
        """The page's address."""
        return f"http://{'localhost' if self.host in ('127.0.0.1', '0.0.0.0') else self.host}:{self.port}/"

    def start(self, timeout: float = 10.0) -> None:
        """Start serving; raises if the server can't start (e.g. the port is taken)."""
        self._thread.start()
        deadline = time.perf_counter() + timeout
        while not self._server.started:
            if not self._thread.is_alive() or time.perf_counter() > deadline:
                raise RuntimeError(f"the web server could not start on {self.host}:{self.port} (port in use?)")
            time.sleep(0.05)
        log.info("web page ready", url=self.url)

    def stop(self) -> None:
        """Stop serving (open pages are disconnected)."""
        self._server.should_exit = True
        self._thread.join(timeout=5.0)
