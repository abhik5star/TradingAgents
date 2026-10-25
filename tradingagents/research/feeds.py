"""Free real-time quotes — and an honest account of where they stop being free.

The dashboard can show a live tape, but only from feeds that actually exist. This
module is the list of those feeds, a standard-library WebSocket client that speaks to
them, and the plumbing that turns a tick stream into candles.

What is genuinely free and keyless
---------------------------------
* **Crypto:** Binance's public trade stream, Coinbase Exchange's `matches` channel and
  Kraken's public ticker. No key, no account, no attribution string. These are real
  exchange feeds, sub-second, and they are what the dashboard shows as *live*.
* **Indian equities:** there is **no** keyless public WebSocket. NSE and BSE do not
  publish one; every "free" Indian real-time feed is either a broker endpoint that
  needs your own credentials (Zerodha Kite, Upstox, Angel One SmartStream) or an
  unofficial scrape that breaks without notice. So the honest options here are:
  a broker WebSocket plugged in through the ``generic`` provider with *your* key
  (``INDIAN_BROKERS`` below says what each one requires), or delayed polling through
  the ``poll`` provider, which is labelled *delayed* everywhere it appears. A tape
  that claims to be live NSE data without a broker account is a tape that is lying.

Everything else is a simulated feed
-----------------------------------
``replay`` re-emits historical bars as if they were ticks. It exists so the dashboard
can be demonstrated (and tested) without a network, and it is labelled ``simulated``
in every payload, panel and API response. Nothing here may be used to produce a
backtest, a paper-trading verdict or a research result: live ticks are display-only,
and :func:`tradingagents.research.paper.run_paper_session` still trades on loaded
bars. Persisting ticks into a CSV and re-running the research is a deliberate act,
not a side effect of watching a chart.
"""

from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import json
import os
import socket
import ssl
import struct
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

# --------------------------------------------------------------------------------------
# What a provider is, and what it costs you
# --------------------------------------------------------------------------------------

WS = "websocket"
POLL = "poll"
REPLAY = "replay"

LIVE = "live"
DELAYED = "delayed"
SIMULATED = "simulated"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Provider:
    """One way to get quotes, with its honest price of admission."""

    name: str
    kind: str
    url: str
    symbols: str
    quality: str
    cost: str
    notes: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind, "url": self.url,
                "symbols": self.symbols, "quality": self.quality, "cost": self.cost,
                "notes": self.notes}


PROVIDERS: dict[str, Provider] = {
    "binance": Provider(
        name="binance", kind=WS, url="wss://stream.binance.com:9443/ws",
        symbols="crypto spot, e.g. BTCUSDT, ETHUSDT, SOLUSDT",
        quality="real-time trade prints",
        cost="free, keyless, no account",
        notes="One connection per stream; use <symbol>@trade. Region-blocked in some "
              "countries, which the dashboard reports as unavailable rather than blank."),
    "coinbase": Provider(
        name="coinbase", kind=WS, url="wss://ws-feed.exchange.coinbase.com",
        symbols="crypto, Coinbase product ids, e.g. BTC-USD, ETH-USD",
        quality="real-time match feed",
        cost="free, keyless, no account",
        notes="Subscribe with {\"type\":\"subscribe\",\"product_ids\":[...],"
              "\"channels\":[\"matches\"]}."),
    "kraken": Provider(
        name="kraken", kind=WS, url="wss://ws.kraken.com/v2",
        symbols="crypto, Kraken pairs, e.g. BTC/USD, ETH/USD",
        quality="real-time trade prints",
        cost="free, keyless, no account",
        notes="v2 API; subscribe to the trade channel per pair."),
    "generic": Provider(
        name="generic", kind=WS,
        url="your broker's endpoint, configured in the feed string",
        symbols="anything your broker streams — Kite, Upstox, Angel One, ...",
        quality="depends on the broker (real-time when they say it is)",
        cost="needs your own API key and a funded account for Indian equities",
        notes="This is the only honest route to live NSE/BSE. Pass the endpoint and "
              "subscribe payload explicitly; no credential is ever stored here. It reads "
              "JSON text frames (Angel One's SmartStream JSON, or any broker that sends "
              "last_price/last_traded_quantity). Binary protocols — Kite's ticker, "
              "Upstox v3 protobuf — need decoding first, so they cannot be read as-is."),
    "poll": Provider(
        name="poll", kind=POLL, url="the existing data layer (Yahoo, delayed)",
        symbols="equities incl. NSE (.NS) and BSE (.BO), indices, FX",
        quality="delayed — typically ~15 minutes for Indian equities",
        cost="free, keyless",
        notes="Not a WebSocket and not a trading feed. Shown as delayed everywhere."),
    "replay": Provider(
        name="replay", kind=REPLAY, url="local bars, re-emitted as ticks",
        symbols="any symbol you have bars for",
        quality="simulated — historical bars played forward",
        cost="free, offline",
        notes="For demos and tests only. Never a market observation."),
}

#: What each Indian broker needs before its WebSocket will talk to you. Recorded so the
#: answer to "free real-time Indian stocks?" is a table rather than a shrug.
INDIAN_BROKERS: dict[str, dict[str, str]] = {
    "kite": {"broker": "Zerodha Kite", "protocol": "binary ticker on wss://ws.kite.trade",
             "needs": "API key + access token from a Kite Connect app; paid API plan",
             "india_free": "no — needs your account and a paid subscription"},
    "upstox": {"broker": "Upstox", "protocol": "protobuf market feed v3",
               "needs": "API key + access token from an Upstox developer app",
               "india_free": "no — needs your account"},
    "angel": {"broker": "Angel One SmartStream", "protocol": "binary/json smart stream",
              "needs": "API key + client code + PIN/TOTP login",
              "india_free": "no — needs your account"},
}

INDIAN_EQUITY_NOTE = (
    "There is no keyless public WebSocket for NSE or BSE. Live Indian quotes need a "
    "broker endpoint and your own API key (Kite, Upstox, Angel One) — see "
    "`tradingagents research feeds`. Without one, the dashboard offers delayed polling "
    "and says so on every panel.")


# --------------------------------------------------------------------------------------
# Ticks and candles
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tick:
    """One price observation, with where it came from."""

    symbol: str
    price: float
    quantity: float
    at: datetime
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "price": self.price, "quantity": self.quantity,
                "at": self.at.isoformat().replace("+00:00", "Z"), "source": self.source}


class BarSeries:
    """Fixed-capacity candles built from ticks, plus the tape's own statistics.

    A tick stream is not a bar: the dashboard needs both, and neither may invent the
    other. ``add`` builds the current minute candle and rolls it when the minute
    changes; the finished candles are what a chart draws.
    """

    def __init__(self, symbol: str, *, seconds: int = 60, capacity: int = 240):
        self.symbol = symbol
        self.seconds = seconds
        self.capacity = capacity
        self.bars: list[dict[str, Any]] = []
        self.last_price: float | None = None
        self.first_price: float | None = None
        self.volume = 0.0
        self.trades = 0
        self.last_at: datetime | None = None
        self._current: dict[str, Any] | None = None

    # ---------- writing ----------

    def add(self, tick: Tick) -> None:
        price = float(tick.price)
        if not price > 0:
            return
        if self.first_price is None:
            self.first_price = price
        self.last_price = price
        self.last_at = tick.at
        self.volume += max(0.0, float(tick.quantity))
        self.trades += 1
        bucket = int(tick.at.timestamp()) // self.seconds * self.seconds
        if self._current is None or bucket != self._current["bucket"]:
            if self._current is not None:
                self._push(self._current)
            self._current = {"bucket": bucket, "t": tick.at, "o": price, "h": price,
                             "l": price, "c": price, "v": max(0.0, float(tick.quantity))}
        else:
            current = self._current
            current["h"] = max(current["h"], price)
            current["l"] = min(current["l"], price)
            current["c"] = price
            current["v"] += max(0.0, float(tick.quantity))

    def _push(self, bar: dict[str, Any]) -> None:
        self.bars.append({
            "time": bar["t"].isoformat().replace("+00:00", "Z"),
            "open": round(bar["o"], 8), "high": round(bar["h"], 8),
            "low": round(bar["l"], 8), "close": round(bar["c"], 8),
            "volume": round(bar["v"], 8),
        })
        if len(self.bars) > self.capacity:
            del self.bars[: len(self.bars) - self.capacity]

    # ---------- reading ----------

    def candles(self) -> list[dict[str, Any]]:
        """Finished candles plus the one being formed, so the chart never lags a minute."""
        bars = list(self.bars)
        if self._current is not None:
            current = self._current
            bars.append({"time": current["t"].isoformat().replace("+00:00", "Z"),
                         "open": round(current["o"], 8), "high": round(current["h"], 8),
                         "low": round(current["l"], 8), "close": round(current["c"], 8),
                         "volume": round(current["v"], 8), "forming": True})
        return bars

    @property
    def change_pct(self) -> float | None:
        if self.first_price is None or self.last_price is None or self.first_price <= 0:
            return None
        return (self.last_price / self.first_price - 1.0) * 100.0

    def summary(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "price": self.last_price,
                "change_pct": None if self.change_pct is None else round(self.change_pct, 4),
                "trades": self.trades, "volume": round(self.volume, 6),
                "last_at": (self.last_at.isoformat().replace("+00:00", "Z")
                            if self.last_at else None),
                "candles": len(self.bars) + (1 if self._current else 0)}


# --------------------------------------------------------------------------------------
# A WebSocket client, standard library only
# --------------------------------------------------------------------------------------

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

#: How many messages a feed may receive without producing a tick before it complains.
_SILENT_AFTER = 20
#: The complaint itself, kept in one place so a later tick can retract it exactly.
_SILENT_WARNING = (f"{_SILENT_AFTER} messages arrived but none carried a recognisable "
                   "price; check the subscribe payload and the field names this provider "
                   "uses")


class WebSocketError(RuntimeError):
    """The socket refused the handshake or spoke something that is not a frame."""


def accept_key(client_key: str) -> str:
    """The Sec-WebSocket-Accept value for a client key (RFC 6455 §4.2.2)."""
    return base64.b64encode(hashlib.sha1((client_key + _GUID).encode()).digest()).decode()


class WebSocket:
    """Minimal client: text frames, ping/pong, close. No extensions, no compression.

    Deliberately small. The alternative is a dependency for three opcodes and a mask,
    and this client is exercised by a real handshake in the tests against a socket we
    speak to ourselves, so the protocol path is not guesswork.
    """

    def __init__(self, url: str, *, headers: dict[str, str] | None = None,
                 timeout: float = 15.0, context: ssl.SSLContext | None = None):
        parsed = urlparse(url)
        if parsed.scheme not in ("ws", "wss"):
            raise WebSocketError(f"not a websocket url: {url!r}")
        self.url = url
        self.secure = parsed.scheme == "wss"
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or (443 if self.secure else 80)
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += "?" + parsed.query
        self.headers = headers or {}
        self.timeout = timeout
        self.context = context
        self._socket: socket.socket | None = None
        self._buffer = b""

    # ---------- lifecycle ----------

    def connect(self) -> None:
        raw = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.secure:
            context = self.context or ssl.create_default_context()
            raw = context.wrap_socket(raw, server_hostname=self.host)
        raw.settimeout(self.timeout)
        self._socket = raw
        key = base64.b64encode(os.urandom(16)).decode()
        lines = [f"GET {self.path} HTTP/1.1", f"Host: {self.host}:{self.port}",
                 "Upgrade: websocket", "Connection: Upgrade", f"Sec-WebSocket-Key: {key}",
                 "Sec-WebSocket-Version: 13"]
        lines += [f"{name}: {value}" for name, value in self.headers.items()]
        raw.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        response = self._read_until_headers()
        status, _, head = response.partition(b"\r\n")
        if b"101" not in status:
            raise WebSocketError(f"handshake refused: {status.decode(errors='replace')}")
        expected = accept_key(key)
        for line in head.decode("latin-1").split("\r\n"):
            name, _, value = line.partition(":")
            if name.strip().lower() == "sec-websocket-accept" and value.strip() != expected:
                raise WebSocketError("handshake failed the accept-key check")

    def _read_until_headers(self) -> bytes:
        assert self._socket is not None
        while b"\r\n\r\n" not in self._buffer:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise WebSocketError("connection closed during the handshake")
            self._buffer += chunk
        head, _, self._buffer = self._buffer.partition(b"\r\n\r\n")
        return head

    def close(self) -> None:
        if self._socket is not None:
            # Closing a dead socket is not an error worth raising.
            with contextlib.suppress(Exception):
                self._send_frame(0x8, b"")
            try:
                self._socket.close()
            finally:
                self._socket = None

    # ---------- frames ----------

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode())

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self._socket is None:
            raise WebSocketError("not connected")
        header = bytearray([0x80 | opcode])
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 1 << 16:
            header.append(0x80 | 126)
            header += struct.pack("!H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", length)
        header += mask
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self._socket.sendall(bytes(header) + masked)

    def recv(self) -> str | None:
        """The next text message, or ``None`` when the peer closed.

        Ping frames are answered inline; continuation frames are joined; a close frame
        is acknowledged and reported as ``None`` so a feed loop can reconnect.
        """
        while True:
            opcode, payload, final = self._read_frame()
            if opcode == 0x8:  # close
                return None
            if opcode == 0x9:  # ping
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:  # pong
                continue
            if opcode in (0x1, 0x0):
                data = payload
                while not final:
                    opcode2, payload2, final = self._read_frame()
                    if opcode2 == 0x8:
                        return None
                    if opcode2 == 0x9:
                        self._send_frame(0xA, payload2)
                        continue
                    data += payload2
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise WebSocketError("binary payload where text was expected") from exc
            if opcode == 0x2:
                return payload.decode("utf-8", "replace")
            raise WebSocketError(f"unexpected opcode {opcode}")

    def _read_frame(self) -> tuple[int, bytes, bool]:
        first = self._read_exact(2)
        fin = bool(first[0] & 0x80)
        opcode = first[0] & 0x0F
        masked = bool(first[1] & 0x80)
        length = first[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        if length > 8 * 1024 * 1024:
            raise WebSocketError(f"frame of {length} bytes is larger than this client accepts")
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        return opcode, payload, fin

    def _read_exact(self, count: int) -> bytes:
        while len(self._buffer) < count:
            assert self._socket is not None
            chunk = self._socket.recv(max(4096, count - len(self._buffer)))
            if not chunk:
                raise WebSocketError("connection closed mid-frame")
            self._buffer += chunk
        data, self._buffer = self._buffer[:count], self._buffer[count:]
        return data


def websocket_server_handshake(request: bytes, sock: socket.socket) -> None:
    """Answer a client handshake. Used by the tests, and by nobody else.

    Having the server side in the same module is what lets the client be tested against
    a real socket instead of a mock: same framing code, opposite direction.
    """
    head = request.decode("latin-1")
    key = ""
    for line in head.split("\r\n"):
        name, _, value = line.partition(":")
        if name.strip().lower() == "sec-websocket-key":
            key = value.strip()
    response = ("HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n")
    sock.sendall(response.encode())


# --------------------------------------------------------------------------------------
# Feed specifications and message parsers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedSpec:
    """One symbol on one provider, with everything needed to subscribe to it."""

    symbol: str
    provider: str = "binance"
    subscribe: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    label: str = ""

    def resolved_url(self) -> str:
        if self.url:
            return self.url
        provider = PROVIDERS.get(self.provider)
        if provider is None:
            raise ValueError(f"unknown provider {self.provider!r}; known: "
                             f"{sorted(PROVIDERS)}")
        if self.provider == "binance":
            return f"{provider.url}/{self.symbol.lower()}@trade"
        return provider.url

    def resolved_subscribe(self) -> str | None:
        if self.subscribe is not None:
            return self.subscribe
        if self.provider == "coinbase":
            return json.dumps({"type": "subscribe", "product_ids": [self.symbol],
                               "channels": ["matches"]})
        if self.provider == "kraken":
            return json.dumps({"method": "subscribe",
                               "params": {"channel": "trade", "symbol": [self.symbol]}})
        return None

    def as_dict(self) -> dict[str, Any]:
        provider = PROVIDERS.get(self.provider)
        return {"symbol": self.symbol, "provider": self.provider,
                "kind": provider.kind if provider else "unknown",
                "url": self.resolved_url(),
                "quality": provider.quality if provider else "unknown",
                "cost": provider.cost if provider else "unknown",
                "label": self.label or self.symbol}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # reject NaN


def _tick(symbol: str, price: Any, quantity: Any, at: Any, source: str) -> Tick | None:
    value = _number(price)
    if value is None or value <= 0:
        return None
    return Tick(symbol=symbol, price=value, quantity=_number(quantity) or 0.0,
                at=_timestamp(at), source=source)


def _timestamp(value: Any) -> datetime:
    """Provider timestamps arrive as seconds, milliseconds or ISO strings."""
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e11:  # milliseconds
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, UTC)
        except (OverflowError, OSError, ValueError):
            return datetime.now(UTC)
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_binance(payload: dict[str, Any], symbol: str) -> Tick | None:
    """Trade events (``e: trade``) and kline events (``e: kline``)."""
    kind = payload.get("e")
    if kind == "trade":
        return _tick(payload.get("s") or symbol, payload.get("p"), payload.get("q"),
                     payload.get("T"), "binance")
    if kind == "kline":
        kline = payload.get("k") or {}
        return _tick(payload.get("s") or symbol, kline.get("c"), kline.get("v"),
                     kline.get("t"), "binance")
    if kind == "24hrTicker" or "c" in payload and "s" in payload:
        return _tick(payload.get("s") or symbol, payload.get("c"), 0, payload.get("E"),
                     "binance")
    return None


def parse_coinbase(payload: dict[str, Any], symbol: str) -> Tick | None:
    if payload.get("type") not in ("match", "last_match", "ticker"):
        return None
    return _tick(payload.get("product_id") or symbol, payload.get("price"),
                 payload.get("size"), payload.get("time"), "coinbase")


def parse_kraken(payload: dict[str, Any], symbol: str) -> Tick | None:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    entry = data[0]
    if not isinstance(entry, dict):
        return None
    return _tick(entry.get("symbol") or symbol, entry.get("price"), entry.get("qty"),
                 entry.get("timestamp"), "kraken")


def parse_generic(payload: dict[str, Any], symbol: str,
                  paths: dict[str, str] | None = None) -> Tick | None:
    """A broker feed whose field names we do not know, described by JSON paths.

    Kite, Upstox and Angel all send a last-traded price and a size under names of their
    own choosing. ``paths`` says where: ``{"price": "last_price", "quantity":
    "last_traded_quantity", "time": "exchange_timestamp"}``.
    """
    paths = paths or {"price": "last_price", "quantity": "last_traded_quantity",
                      "time": "exchange_timestamp", "symbol": "tradingsymbol"}
    def lookup(path: str) -> Any:
        value: Any = payload
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value
    return _tick(lookup(paths.get("symbol", "symbol")) or symbol, lookup(paths["price"]),
                 lookup(paths.get("quantity", "quantity")),
                 lookup(paths.get("time", "time")), "broker")


PARSERS: dict[str, Callable[[dict[str, Any], str], Tick | None]] = {
    "binance": parse_binance,
    "coinbase": parse_coinbase,
    "kraken": parse_kraken,
}


def parse_message(provider: str, raw: str, symbol: str) -> list[Tick]:
    """Every tick in one provider message. Unknown or malformed messages yield ``[]``.

    A live feed sends heartbeats, subscription confirmations and occasional binary
    frames; a parser that raised on those would drop the connection every few seconds.
    """
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    parser = PARSERS.get(provider, parse_generic)
    try:
        tick = parser(payload, symbol)
    except Exception:  # one odd message must not end the stream
        return []
    return [tick] if tick else []


# --------------------------------------------------------------------------------------
# Feeds
# --------------------------------------------------------------------------------------


class Feed:
    """A background producer of ticks into one :class:`BarSeries`."""

    quality = LIVE

    def __init__(self, spec: FeedSpec, series: BarSeries, *, reconnect: bool = True):
        self.spec = spec
        self.series = series
        self.reconnect = reconnect
        self.status = UNAVAILABLE
        self.error = ""
        self.connected = False
        self._stop = threading.Event()

    # ---------- lifecycle ----------

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def run(self) -> None:  # pragma: no cover - the loop itself is exercised via _pump
        delay = 1.0
        while not self.stopped:
            try:
                self._session()
                delay = 1.0
            except Exception as exc:  # an unreachable feed is a status, not a crash
                self.status = UNAVAILABLE
                # Keep a field-name complaint: a disconnect should not erase the reason
                # the feed was already useless.
                prior = self.error if self.error == _SILENT_WARNING else ""
                self.error = f"{type(exc).__name__}: {exc}"[:300]
                if prior:
                    self.error = f"{self.error} ({prior})"[:300]
                self.connected = False
            if not self.reconnect or self.stopped:
                return
            if self._stop.wait(delay):
                return
            delay = min(delay * 2, 30.0)

    def _session(self) -> None:
        transport = self.connect()
        try:
            self.status = self.quality
            self.connected = True
            self.error = ""
            self.subscribe(transport)
            messages = 0
            ticks = 0
            while not self.stopped:
                message = transport.recv()
                if message is None:
                    raise ConnectionError("the feed closed the connection")
                messages += 1
                for tick in parse_message(self.spec.provider, message, self.spec.symbol):
                    self.series.add(tick)
                    ticks += 1
                if messages >= _SILENT_AFTER and not ticks:
                    # Connected, chatting, and producing nothing: almost always a field
                    # name mismatch on a broker feed. Say so instead of showing a flat line.
                    self.error = _SILENT_WARNING
                elif ticks and self.error == _SILENT_WARNING:
                    self.error = ""
        finally:
            self.connected = False
            # Closing twice is harmless.
            with contextlib.suppress(Exception):
                transport.close()

    # ---------- pieces a subclass replaces ----------

    def connect(self) -> Any:
        raise NotImplementedError

    def subscribe(self, transport: Any) -> None:
        message = self.spec.resolved_subscribe()
        if message:
            transport.send_text(message)

    def state(self) -> dict[str, Any]:
        return {"symbol": self.spec.symbol, "provider": self.spec.provider,
                "quality": self.quality, "status": self.status if self.connected
                else (self.status if self.error else UNAVAILABLE),
                "connected": self.connected, "error": self.error,
                "spec": self.spec.as_dict(), "series": self.series.summary()}


class WebSocketFeed(Feed):
    """A provider's public WebSocket stream."""

    def __init__(self, spec: FeedSpec, series: BarSeries, *, transport_factory: Any = None,
                 reconnect: bool = True):
        super().__init__(spec, series, reconnect=reconnect)
        self._transport_factory = transport_factory

    def connect(self) -> Any:
        if self._transport_factory is not None:
            return self._transport_factory(self.spec.resolved_url())
        socket_ = WebSocket(self.spec.resolved_url(), headers=self.spec.headers)
        socket_.connect()
        return socket_


class ReplayFeed(Feed):
    """Historical bars re-emitted as ticks. Simulated, and labelled that way."""

    quality = SIMULATED

    def __init__(self, spec: FeedSpec, series: BarSeries,
                 bars: Iterable[dict[str, Any]], *, per_second: float = 8.0,
                 loop: bool = True, clock: Callable[[], float] = time.monotonic,
                 step_seconds: float = 60.0, start_at: datetime | None = None):
        super().__init__(spec, series, reconnect=False)
        self.bars = list(bars)
        self.per_second = max(0.5, per_second)
        self.loop = loop
        self.clock = clock
        # A replay runs on a compressed clock: each replayed bar advances the tick
        # timestamp by a candle length, so one bar becomes one candle instead of
        # sixty ticks collapsing into the current minute. The clock only moves forward,
        # even when the bar list loops, so candles never arrive out of order.
        self.step_seconds = max(0.0, float(step_seconds))
        self._at = start_at
        self.started = 0
        self.cursor = 0

    def run(self) -> None:
        if not self.bars:
            self.status = UNAVAILABLE
            self.error = "no bars to replay"
            return
        self.status = SIMULATED
        self.connected = True
        try:
            self._pump()
        finally:
            # A stopped replay is not a live feed any more; leaving ``connected`` set
            # would make the dashboard claim a stream that has already ended.
            self.connected = False

    def _pump(self) -> None:
        interval = 1.0 / self.per_second
        next_at = self.clock()
        while not self.stopped:
            if next_at > self.clock():
                if self._stop.wait(min(0.25, next_at - self.clock())):
                    break
                continue
            bar = self.bars[self.cursor % len(self.bars)]
            self.cursor += 1
            price = _number(bar.get("close")) or 0.0
            at = datetime.now(UTC) if self._at is None else self._at
            if price > 0:
                self.series.add(Tick(symbol=self.spec.symbol, price=price,
                                     quantity=_number(bar.get("volume")) or 0.0,
                                     at=at, source="replay"))
                self.started += 1
            self._at = at + timedelta(seconds=self.step_seconds)
            next_at += interval
            if self.cursor >= len(self.bars):
                if not self.loop:
                    self.status = SIMULATED
                    self.error = "replay finished"
                    break
                self.cursor = 0

    def _session(self) -> None:  # replay does not reconnect; run() is the whole loop
        return

    def state(self) -> dict[str, Any]:
        state = super().state()
        state["status"] = SIMULATED if self.connected else state["status"]
        state["note"] = ("simulated replay of historical bars on a compressed clock — "
                         "not a market observation")
        return state


class PollFeed(Feed):
    """Delayed quotes from a polling fetcher (the data layer), labelled as delayed."""

    quality = DELAYED

    def __init__(self, spec: FeedSpec, series: BarSeries,
                 fetch: Callable[[str], float | None], *, interval: float = 60.0):
        super().__init__(spec, series, reconnect=False)
        self.fetch = fetch
        self.interval = max(5.0, interval)

    def run(self) -> None:
        while not self.stopped:
            try:
                price = self.fetch(self.spec.symbol)
                if price is not None and price > 0:
                    self.series.add(Tick(symbol=self.spec.symbol, price=float(price),
                                         quantity=0.0, at=datetime.now(UTC), source="poll"))
                    self.status = DELAYED
                    self.error = ""
                else:
                    self.status = UNAVAILABLE
                    self.error = "the data layer returned no price"
            except Exception as exc:
                self.status = UNAVAILABLE
                self.error = f"{type(exc).__name__}: {exc}"[:300]
            if self._stop.wait(self.interval):
                return

    def _session(self) -> None:
        return

    def state(self) -> dict[str, Any]:
        state = super().state()
        state["note"] = "delayed quotes from the data layer — not a trading feed"
        return state


# --------------------------------------------------------------------------------------
# The hub
# --------------------------------------------------------------------------------------


class FeedHub:
    """Every configured feed, running in its own thread, read under one lock."""

    def __init__(self) -> None:
        self._feeds: list[Feed] = []
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.started_at: datetime | None = None

    # ---------- building ----------

    def add(self, feed: Feed) -> Feed:
        self._feeds.append(feed)
        return feed

    @classmethod
    def from_specs(cls, specs: Iterable[FeedSpec], *, bar_factory: Callable[[str], BarSeries]
                   | None = None, replay_bars: dict[str, list[dict[str, Any]]] | None = None,
                   per_second: float = 8.0,
                   fetcher: Callable[[str], float | None] | None = None,
                   transport_factory: Any = None) -> FeedHub:
        """Build a hub from ``provider:SYMBOL`` specifications.

        ``binance:BTCUSDT``, ``coinbase:BTC-USD``, ``kraken:BTC/USD``,
        ``poll:RELIANCE.NS``, ``replay:SYNTH``. Unknown providers are rejected here
        rather than silently ignored, because a typo that produces an empty dashboard is
        worse than an error.
        """
        hub = cls()
        for spec in specs:
            if spec.provider not in PROVIDERS:
                raise ValueError(f"unknown provider {spec.provider!r}; known: "
                                 f"{sorted(PROVIDERS)}")
            if spec.provider == "replay":
                bar_factory = bar_factory or (lambda symbol: BarSeries(symbol))
                series = bar_factory(spec.symbol)
                bars = (replay_bars or {}).get(spec.symbol, [])
                if not bars:
                    feed: Feed = ReplayFeed(spec, series, [], per_second=per_second)
                else:
                    feed = ReplayFeed(spec, series, bars, per_second=per_second)
                hub.add(feed)
                continue
            kind = PROVIDERS[spec.provider].kind
            series = (bar_factory or (lambda symbol: BarSeries(symbol)))(spec.symbol)
            if kind == POLL:
                if fetcher is None:
                    raise ValueError("a poll feed needs a fetcher")
                hub.add(PollFeed(spec, series, fetcher))
            else:
                hub.add(WebSocketFeed(spec, series, transport_factory=transport_factory))
        return hub

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._threads:
            return
        self.started_at = datetime.now(UTC)
        for feed in self._feeds:
            thread = threading.Thread(target=feed.run, name=f"feed-{feed.spec.symbol}",
                                      daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        for feed in self._feeds:
            feed.stop()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()

    # ---------- reading ----------

    def state(self) -> dict[str, Any]:
        """A display snapshot: what each feed is, whether it is up, and its candles."""
        with self._lock:
            feeds = [feed.state() for feed in self._feeds]
            candles = {feed.spec.symbol: feed.series.candles() for feed in self._feeds}
        for entry in feeds:
            entry["candles"] = candles.get(entry["symbol"], [])
        return {"generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "started_at": (self.started_at.isoformat().replace("+00:00", "Z")
                               if self.started_at else None),
                "feeds": feeds,
                "indian_equity_note": INDIAN_EQUITY_NOTE,
                "disclaimer": ("Live feeds are display-only. Research, paper trading and "
                               "backtests run on loaded bars, never on this tape.")}

    def symbols(self) -> list[str]:
        return [feed.spec.symbol for feed in self._feeds]

    def specs(self) -> list[FeedSpec]:
        return [feed.spec for feed in self._feeds]


def parse_feed_specs(items: Iterable[str]) -> list[FeedSpec]:
    """``provider:SYMBOL`` strings, as typed on the command line."""
    specs: list[FeedSpec] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        provider, _, symbol = text.partition(":")
        if not symbol:
            provider, symbol = ("binance", provider)
        provider = provider.strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r} in {text!r}; known: "
                             f"{sorted(PROVIDERS)}")
        specs.append(FeedSpec(symbol=symbol.strip(), provider=provider))
    return specs


def attach_endpoints(specs: Iterable[FeedSpec], urls: Iterable[str] | None = None,
                    subscribes: Iterable[str] | None = None) -> list[FeedSpec]:
    """Give ``generic`` specs the endpoint and subscribe payload the user typed.

    A broker stream cannot be guessed: the URL, the message and the credential are all
    broker-specific. The command line therefore takes them next to the feed, in order:

        --feed generic:RELIANCE --feed-url wss://ws.kite.trade \
            --feed-subscribe '{"a":"subscribe","v":[256265]}'

    Every generic spec must end up with a URL. Silently substituting a public endpoint
    for a broker one would turn a connection error into a lie.
    """
    pending_urls = list(urls or [])
    pending_subs = list(subscribes or [])
    resolved: list[FeedSpec] = []
    for spec in specs:
        if spec.provider != "generic":
            resolved.append(spec)
            continue
        url = spec.url or (pending_urls.pop(0) if pending_urls else None)
        subscribe = spec.subscribe
        if subscribe is None and pending_subs:
            subscribe = pending_subs.pop(0)
        if not url:
            raise ValueError(
                f"feed {spec.symbol!r} uses the generic (broker) provider but no URL was "
                "given: pass --feed-url with your broker's WebSocket endpoint")
        resolved.append(FeedSpec(symbol=spec.symbol, provider=spec.provider,
                                 subscribe=subscribe, url=url, headers=dict(spec.headers),
                                 label=spec.label))
    if pending_urls or pending_subs:
        raise ValueError("--feed-url/--feed-subscribe given for a feed that does not use "
                         "them; a generic feed needs one URL per spec")
    return resolved


_CSV_COLUMNS = {
    "time": ("time", "date", "datetime", "timestamp", "ts"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "adj close", "adj_close"),
    "volume": ("volume", "v", "vol"),
}


def replay_bars_from_csv(path: str | os.PathLike[str], *, limit: int = 480,
                         tail: bool = True) -> list[dict[str, Any]]:
    """Bars from a CSV, for a ``replay:`` feed.

    Deliberately forgiving about column names and deliberately strict about the one
    thing that matters: a row without a usable ``close`` is dropped, not filled. The
    result is labelled simulated by :class:`ReplayFeed`; this function does not pretend
    to know where the file came from.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: no header row")
        index = {}
        for field, candidates in _CSV_COLUMNS.items():
            for name in reader.fieldnames:
                if name and name.strip().lower() in candidates:
                    index[field] = name
                    break
        if "close" not in index:
            raise ValueError(f"{path}: no close column (looked for {_CSV_COLUMNS['close']})")
        bars: list[dict[str, Any]] = []
        for row in reader:
            close = _number(row.get(index["close"]))
            if close is None or close <= 0:
                continue
            open_ = _number(row.get(index["open"])) if "open" in index else None
            high = _number(row.get(index["high"])) if "high" in index else None
            low = _number(row.get(index["low"])) if "low" in index else None
            bars.append({
                "time": str(row.get(index.get("time", ""), "") or ""),
                "open": open_ if open_ is not None else close,
                "high": high if high is not None else close,
                "low": low if low is not None else close,
                "close": close,
                "volume": _number(row.get(index["volume"])) if "volume" in index else 0.0,
            })
    if not bars:
        raise ValueError(f"{path}: no usable rows")
    if limit and len(bars) > limit:
        bars = bars[-limit:] if tail else bars[:limit]
    return bars


def replay_bars_for(symbols: Iterable[str], files: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    """Map ``SYMBOL=PATH`` pairs onto the symbols that actually replay them."""
    wanted = set(symbols)
    bars: dict[str, list[dict[str, Any]]] = {}
    for symbol, path in files.items():
        if symbol not in wanted:
            raise ValueError(f"--replay-csv was given for {symbol!r}, which is not a "
                             f"replay feed (replay feeds: {sorted(wanted) or 'none'})")
        bars[symbol] = replay_bars_from_csv(path)
    missing = sorted(wanted - set(bars))
    if missing:
        raise ValueError(f"replay feed(s) {missing} need bars: pass --replay-csv "
                         "SYMBOL=PATH for each one")
    return bars
