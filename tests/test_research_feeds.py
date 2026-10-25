"""The free feeds: real protocol, real parsers, and labels that survive the trip.

The interesting tests here talk to a socket rather than a mock. ``websocket_server_
handshake`` lives beside the client in the module under test, so the handshake, the
masking, the ping reply and the close are all exercised against bytes on a socket pair
instead of against an assertion about which method was called.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.research import feeds
from tradingagents.research.feeds import (
    DELAYED,
    LIVE,
    SIMULATED,
    UNAVAILABLE,
    BarSeries,
    FeedHub,
    FeedSpec,
    PollFeed,
    ReplayFeed,
    Tick,
    WebSocket,
    WebSocketError,
    WebSocketFeed,
    accept_key,
    attach_endpoints,
    parse_feed_specs,
    parse_message,
    replay_bars_for,
    replay_bars_from_csv,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class FakeTransport:
    """A feed peer in memory: yields queued messages, records what was sent."""

    def __init__(self, messages: list[str] | None = None, *, raise_on_recv: bool = False):
        self.messages = list(messages or [])
        self.raise_on_recv = raise_on_recv
        self.sent: list[str] = []
        self.closed = False

    def send_text(self, text: str) -> None:
        self.sent.append(text)

    def recv(self) -> str | None:
        if self.raise_on_recv:
            raise ConnectionResetError("peer went away")
        return self.messages.pop(0) if self.messages else None

    def close(self) -> None:
        self.closed = True


def _server_frame(opcode: int, payload: bytes, *, final: bool = True) -> bytes:
    """An unmasked server-to-client frame (servers do not mask)."""
    head = bytearray([(0x80 if final else 0x00) | opcode])
    length = len(payload)
    if length < 126:
        head.append(length)
    elif length < 1 << 16:
        head.append(126)
        head += struct.pack("!H", length)
    else:
        head.append(127)
        head += struct.pack("!Q", length)
    return bytes(head) + payload


def _read_client_frame(sock: socket.socket) -> tuple[int, bytes, bool]:
    """Read one client frame and unmask it, checking the mask bit on the way."""
    first = sock.recv(2)
    assert len(first) == 2
    fin = bool(first[0] & 0x80)
    opcode = first[0] & 0x0F
    masked = bool(first[1] & 0x80)
    length = first[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    assert masked, "a client frame must be masked"
    mask = _recv_exact(sock, 4)
    payload = _recv_exact(sock, length) if length else b""
    return opcode, bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload)), fin


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    buffer = b""
    while len(buffer) < count:
        chunk = sock.recv(count - len(buffer))
        if not chunk:
            raise AssertionError("socket closed early")
        buffer += chunk
    return buffer


@pytest.fixture()
def handshake(monkeypatch):
    """A connected client whose handshake was answered by the module's own server side.

    ``socket.create_connection`` is replaced with one end of a socket pair, so the
    client still builds and parses real HTTP bytes and the transport is a real socket.
    """
    sockets: list[socket.socket] = []

    def make(headers: dict[str, str] | None = None, path: str = "/stream"):
        client_sock, server_sock = socket.socketpair()
        server_sock.settimeout(5.0)
        sockets.extend([client_sock, server_sock])
        monkeypatch.setattr(feeds.socket, "create_connection", lambda *a, **k: client_sock)
        client = WebSocket(f"ws://feed.test{path}", headers=headers, timeout=5.0)
        seen: dict[str, bytes] = {}

        def serve() -> None:
            buffer = b""
            while b"\r\n\r\n" not in buffer:
                chunk = server_sock.recv(4096)
                if not chunk:
                    return
                buffer += chunk
            seen["head"] = buffer
            feeds.websocket_server_handshake(buffer, server_sock)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        client.connect()
        thread.join(timeout=5)
        return client, server_sock, seen.get("head", b"")

    yield make
    for sock in sockets:
        sock.close()


# --------------------------------------------------------------------------------------
# The WebSocket client
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_accept_key_matches_the_rfc_example():
    # RFC 6455 section 1.3: this key must produce this accept value.
    assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


@pytest.mark.unit
def test_the_handshake_request_uses_the_path_and_the_custom_headers(handshake):
    client, _server, request = handshake({"X-API-Key": "secret-value"}, path="/stream?x=1")
    try:
        head = request.decode("latin-1")
        assert head.startswith("GET /stream?x=1 HTTP/1.1")
        assert "Upgrade: websocket" in head
        assert "Sec-WebSocket-Version: 13" in head
        assert "x-api-key: secret-value" in head.lower()
        assert "Sec-WebSocket-Key:" in head
    finally:
        client.close()


@pytest.mark.unit
def test_a_message_arrives_and_a_fragmented_one_is_joined(handshake):
    client, server, _request = handshake()
    try:
        document = json.dumps({"p": "1", "q": "2"}).encode()
        server.sendall(_server_frame(0x1, document[:5], final=False))
        server.sendall(_server_frame(0x0, document[5:], final=True))
        assert json.loads(client.recv()) == {"p": "1", "q": "2"}
    finally:
        client.close()


@pytest.mark.unit
def test_a_ping_is_answered_with_a_masked_pong(handshake):
    client, server, _request = handshake()
    try:
        # The client answers pings while it is reading, so a real message follows the
        # ping and gives recv() something to return.
        server.sendall(_server_frame(0x9, b"keepalive"))
        server.sendall(_server_frame(0x1, b'{"e":"ping"}'))
        received: dict[str, str | None] = {}
        reader = threading.Thread(target=lambda: received.update(msg=client.recv()),
                                  daemon=True)
        reader.start()
        opcode, payload, _final = _read_client_frame(server)
        reader.join(timeout=5)
        assert opcode == 0xA
        assert payload == b"keepalive"
        assert received["msg"] == '{"e":"ping"}'
    finally:
        client.close()


@pytest.mark.unit
def test_sending_text_masks_the_payload(handshake):
    client, server, _request = handshake()
    try:
        client.send_text('{"type":"subscribe","product_ids":["BTC-USD"]}')
        opcode, payload, final = _read_client_frame(server)
        assert opcode == 0x1 and final
        assert json.loads(payload)["product_ids"] == ["BTC-USD"]
    finally:
        client.close()


@pytest.mark.unit
def test_a_close_frame_ends_the_stream_instead_of_raising(handshake):
    client, server, _request = handshake()
    try:
        server.sendall(_server_frame(0x8, b""))
        assert client.recv() is None
    finally:
        client.close()


@pytest.mark.unit
def test_a_refused_handshake_is_an_error_not_a_hang(monkeypatch):
    left, right = socket.socketpair()
    left.settimeout(5.0)
    right.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
    monkeypatch.setattr(feeds.socket, "create_connection", lambda *a, **k: left)
    client = WebSocket("ws://feed.test/stream", timeout=2.0)
    with pytest.raises(WebSocketError, match="handshake refused"):
        client.connect()
    right.close()
    left.close()


@pytest.mark.unit
def test_a_bad_accept_key_is_rejected(monkeypatch):
    left, right = socket.socketpair()
    left.settimeout(5.0)
    right.settimeout(5.0)

    def serve() -> None:
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            buffer += right.recv(4096)
        right.sendall(b"HTTP/1.1 101 Switching Protocols\r\n"
                      b"Sec-WebSocket-Accept: not-the-right-value\r\n\r\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setattr(feeds.socket, "create_connection", lambda *a, **k: left)
    client = WebSocket("ws://feed.test/stream", timeout=2.0)
    with pytest.raises(WebSocketError, match="accept-key"):
        client.connect()
    thread.join(timeout=5)
    right.close()
    left.close()


@pytest.mark.unit
def test_a_non_websocket_url_is_refused():
    with pytest.raises(WebSocketError, match="not a websocket url"):
        WebSocket("https://example.invalid/stream")


# --------------------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_each_provider_resolves_the_endpoint_it_actually_publishes():
    specs = {spec.provider: spec for spec in parse_feed_specs(
        ["binance:BTCUSDT", "coinbase:BTC-USD", "kraken:BTC/USD", "poll:RELIANCE.NS"])}
    assert specs["binance"].resolved_url() == "wss://stream.binance.com:9443/ws/btcusdt@trade"
    assert specs["coinbase"].resolved_url() == "wss://ws-feed.exchange.coinbase.com"
    assert json.loads(specs["coinbase"].resolved_subscribe()) == {
        "type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["matches"]}
    assert specs["kraken"].resolved_url() == "wss://ws.kraken.com/v2"
    assert json.loads(specs["kraken"].resolved_subscribe())["params"]["symbol"] == ["BTC/USD"]
    assert specs["poll"].resolved_url().startswith("the existing data layer")


@pytest.mark.unit
def test_a_bare_symbol_defaults_to_binance_and_a_typo_is_refused():
    assert parse_feed_specs(["ETHUSDT"])[0].provider == "binance"
    with pytest.raises(ValueError, match="unknown provider"):
        parse_feed_specs(["nse:RELIANCE"])


@pytest.mark.unit
def test_a_broker_feed_needs_its_own_endpoint():
    specs = parse_feed_specs(["generic:RELIANCE"])
    with pytest.raises(ValueError, match="--feed-url"):
        attach_endpoints(specs, [], [])
    resolved = attach_endpoints(specs, ["wss://ws.kite.trade"], ['{"a":"subscribe"}'])
    assert resolved[0].resolved_url() == "wss://ws.kite.trade"
    assert resolved[0].resolved_subscribe() == '{"a":"subscribe"}'


@pytest.mark.unit
def test_an_endpoint_given_for_a_feed_that_cannot_use_it_is_refused():
    specs = parse_feed_specs(["binance:BTCUSDT"])
    with pytest.raises(ValueError, match="does not use"):
        attach_endpoints(specs, ["wss://ws.kite.trade"], [])


# --------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_three_keyless_crypto_providers_parse_into_ticks():
    binance = parse_message("binance", json.dumps(
        {"e": "trade", "s": "BTCUSDT", "p": "64321.25", "q": "0.016",
         "T": 1_710_000_000_000}), "BTCUSDT")
    assert len(binance) == 1
    assert binance[0].price == 64321.25 and binance[0].quantity == 0.016
    assert binance[0].at.tzinfo is not None
    assert binance[0].at.year == 2024

    coinbase = parse_message("coinbase", json.dumps(
        {"type": "match", "product_id": "BTC-USD", "price": "64300.5", "size": "0.02",
         "time": "2024-03-09T16:00:00.000000Z"}), "BTC-USD")
    assert coinbase and coinbase[0].price == 64300.5

    kraken = parse_message("kraken", json.dumps(
        {"channel": "trade", "type": "update",
         "data": [{"symbol": "BTC/USD", "price": 64200.1, "qty": 0.5,
                   "timestamp": "2024-03-09T16:00:00.000000Z"}]}), "BTC/USD")
    assert kraken and kraken[0].symbol == "BTC/USD" and kraken[0].price == 64200.1


@pytest.mark.unit
def test_a_broker_payload_is_read_through_declared_paths():
    payload = {"data": {"last_price": 2875.4, "last_traded_quantity": 12,
                        "exchange_timestamp": "2024-03-09T10:00:00+05:30"}}
    tick = feeds.parse_generic(payload, "RELIANCE",
                               {"price": "data.last_price",
                                "quantity": "data.last_traded_quantity",
                                "time": "data.exchange_timestamp"})
    assert tick is not None
    assert tick.price == 2875.4 and tick.quantity == 12.0
    assert tick.at.utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.unit
@pytest.mark.parametrize("raw", [
    "not json at all",
    "[]",
    json.dumps({"type": "subscriptions", "channels": []}),
    json.dumps({"e": "error", "m": "Unknown symbol"}),
    json.dumps({"e": "trade", "p": "0"}),
    json.dumps({"e": "trade", "p": "not-a-number"}),
])
def test_a_message_that_is_not_a_trade_yields_no_tick(raw):
    assert parse_message("binance", raw, "BTCUSDT") == []


@pytest.mark.unit
def test_a_nan_price_is_rejected_rather_than_charted():
    assert feeds.parse_binance({"e": "trade", "p": "nan"}, "BTCUSDT") is None
    assert feeds.parse_binance({"e": "trade", "p": "-5"}, "BTCUSDT") is None


# --------------------------------------------------------------------------------------
# Candles
# --------------------------------------------------------------------------------------


def _tick(price: float, *, second: float = 0, symbol: str = "SYNTH") -> Tick:
    return Tick(symbol=symbol, price=price, quantity=1.0,
                at=datetime(2026, 1, 1, 9, 30, tzinfo=UTC) + timedelta(seconds=second),
                source="test")


@pytest.mark.unit
def test_ticks_become_a_candle_and_a_new_minute_rolls_it():
    series = BarSeries("SYNTH", seconds=60)
    series.add(_tick(100))
    series.add(_tick(104, second=10))
    series.add(_tick(98, second=20))
    forming = series.candles()
    assert len(forming) == 1 and forming[0]["forming"] is True
    assert (forming[0]["open"], forming[0]["high"], forming[0]["low"], forming[0]["close"]) == (
        100, 104, 98, 98)

    series.add(_tick(101, second=61))
    candles = series.candles()
    assert len(candles) == 2
    assert candles[0].get("forming") is not True  # the finished minute, now closed
    assert candles[0]["close"] == 98
    assert candles[1]["forming"] is True and candles[1]["open"] == 101
    assert series.summary()["candles"] == 2
    assert series.change_pct == pytest.approx(1.0)


@pytest.mark.unit
def test_the_series_keeps_a_bounded_window_and_counts_the_tape():
    series = BarSeries("SYNTH", seconds=1, capacity=5)
    for index in range(12):
        series.add(_tick(100 + index, second=index))
    candles = series.candles()
    assert len(candles) == 6  # five finished plus the forming one
    assert candles[-1]["close"] == 111
    summary = series.summary()
    assert summary["trades"] == 12 and summary["volume"] == 12.0
    assert summary["last_at"].endswith("Z")


# --------------------------------------------------------------------------------------
# Feeds
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_a_websocket_feed_subscribes_and_stores_parsed_ticks():
    transport = FakeTransport([
        json.dumps({"type": "subscriptions", "channels": []}),
        json.dumps({"type": "match", "product_id": "BTC-USD", "price": "100.5",
                    "size": "2", "time": "2026-01-01T00:00:00Z"}),
        json.dumps({"type": "match", "product_id": "BTC-USD", "price": "101.5",
                    "size": "1", "time": "2026-01-01T00:01:00Z"}),
    ])
    spec = FeedSpec(symbol="BTC-USD", provider="coinbase")
    series = BarSeries("BTC-USD")
    feed = WebSocketFeed(spec, series, transport_factory=lambda url: transport,
                        reconnect=False)
    feed.run()  # ends by itself when the queue is empty (recv -> None)

    assert json.loads(transport.sent[0])["product_ids"] == ["BTC-USD"]
    assert transport.closed is True
    assert series.last_price == 101.5
    assert [candle["close"] for candle in series.candles()] == [100.5, 101.5]
    state = feed.state()
    assert state["quality"] == LIVE
    assert state["connected"] is False  # the session is over
    assert "closed the connection" in state["error"]


@pytest.mark.unit
def test_an_unreachable_feed_reports_unavailable_instead_of_crashing():
    def factory(url):
        raise OSError("connection refused")

    feed = WebSocketFeed(FeedSpec(symbol="BTCUSDT", provider="binance"),
                        BarSeries("BTCUSDT"), transport_factory=factory, reconnect=False)
    feed.run()
    state = feed.state()
    assert state["status"] == UNAVAILABLE
    assert state["connected"] is False
    assert "connection refused" in state["error"]


@pytest.mark.unit
def test_a_replay_feed_turns_each_bar_into_one_candle_and_says_it_is_simulated():
    bars = [{"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i,
             "volume": 5} for i in range(6)]
    series = BarSeries("SYNTH")
    feed = ReplayFeed(FeedSpec(symbol="SYNTH", provider="replay"), series, bars,
                      per_second=400, loop=False,
                      start_at=datetime(2026, 1, 1, tzinfo=UTC))
    thread = threading.Thread(target=feed.run, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()

    candles = series.candles()
    assert [candle["close"] for candle in candles] == [bar["close"] for bar in bars]
    assert all(candle.get("forming") is not True for candle in candles[:-1])
    state = feed.state()
    assert state["quality"] == SIMULATED
    assert "simulated" in state["note"]
    assert state["status"] == SIMULATED
    # Determinism: the same bars replay into the same candles.
    second = BarSeries("SYNTH")
    ReplayFeed(FeedSpec(symbol="SYNTH", provider="replay"), second, bars,
               per_second=400, loop=False,
               start_at=datetime(2026, 1, 1, tzinfo=UTC)).run()
    assert [c["close"] for c in second.candles()] == [c["close"] for c in candles]


@pytest.mark.unit
def test_a_replay_feed_with_no_bars_says_so():
    series = BarSeries("SYNTH")
    feed = ReplayFeed(FeedSpec(symbol="SYNTH", provider="replay"), series, [])
    feed.run()
    assert feed.state()["status"] == UNAVAILABLE
    assert "no bars" in feed.state()["error"]


@pytest.mark.unit
def test_a_poll_feed_is_labelled_delayed_and_reports_a_vendor_failure():
    series = BarSeries("RELIANCE.NS")
    feed = PollFeed(FeedSpec(symbol="RELIANCE.NS", provider="poll"), series,
                    lambda symbol: 2875.4, interval=60.0)
    thread = threading.Thread(target=feed.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while series.last_price is None and time.monotonic() < deadline:
        time.sleep(0.01)
    feed.stop()
    thread.join(timeout=5)
    assert series.last_price == 2875.4
    state = feed.state()
    assert state["quality"] == DELAYED
    assert "delayed" in state["note"]

    def broken(symbol):
        raise RuntimeError("vendor rate limit")

    failing = PollFeed(FeedSpec(symbol="RELIANCE.NS", provider="poll"), BarSeries("RELIANCE.NS"),
                       broken)
    thread = threading.Thread(target=failing.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not failing.error and time.monotonic() < deadline:
        time.sleep(0.01)
    failing.stop()
    thread.join(timeout=5)
    assert failing.state()["status"] == UNAVAILABLE
    assert "rate limit" in failing.state()["error"]


# --------------------------------------------------------------------------------------
# The hub
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_hub_snapshot_carries_quality_and_the_indian_equity_truth():
    hub = FeedHub.from_specs(
        parse_feed_specs(["replay:SYNTH"]),
        replay_bars={"SYNTH": [{"open": 1, "high": 2, "low": 1, "close": 1.5}]})
    state = hub.state()
    assert state["feeds"][0]["quality"] == SIMULATED
    assert state["feeds"][0]["candles"] == []
    assert "no keyless public WebSocket" in state["indian_equity_note"]
    assert "display-only" in state["disclaimer"]
    assert state["started_at"] is None
    assert hub.symbols() == ["SYNTH"]
    assert [spec.provider for spec in hub.specs()] == ["replay"]


@pytest.mark.unit
def test_an_empty_hub_is_still_a_valid_answer():
    state = FeedHub().state()
    assert state["feeds"] == []
    assert state["indian_equity_note"]
    assert state["disclaimer"]


@pytest.mark.unit
def test_the_hub_runs_feeds_on_their_own_threads_and_stops_them():
    bars = [{"open": 10 + i, "high": 11 + i, "low": 9 + i, "close": 10.5 + i} for i in range(20)]
    hub = FeedHub.from_specs(parse_feed_specs(["replay:SYNTH"]),
                             replay_bars={"SYNTH": bars}, per_second=400)
    hub.start()
    try:
        deadline = time.monotonic() + 5
        while not hub.state()["feeds"][0]["candles"] and time.monotonic() < deadline:
            time.sleep(0.01)
        state = hub.state()
        assert state["started_at"] is not None
        assert state["feeds"][0]["connected"] is True
        assert state["feeds"][0]["candles"]
    finally:
        hub.stop()
    assert hub.state()["feeds"][0]["connected"] is False


@pytest.mark.unit
def test_a_provider_the_registry_does_not_know_is_refused_by_the_hub():
    with pytest.raises(ValueError, match="unknown provider"):
        FeedHub.from_specs([FeedSpec(symbol="X", provider="nonsense")])


@pytest.mark.unit
def test_a_poll_feed_cannot_be_built_without_a_fetcher():
    with pytest.raises(ValueError, match="needs a fetcher"):
        FeedHub.from_specs(parse_feed_specs(["poll:RELIANCE.NS"]))


# --------------------------------------------------------------------------------------
# Replay bars from a file the user already has
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_replay_bars_are_read_from_a_csv_with_any_reasonable_header(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text("Date,Open,High,Low,Close,Volume\n"
                    "2026-01-01,10,11,9,10.5,100\n"
                    "2026-01-02,10.5,12,10,11.5,120\n"
                    "2026-01-03,11.5,12,11,,\n", encoding="utf-8")
    bars = replay_bars_from_csv(path)
    assert [bar["close"] for bar in bars] == [10.5, 11.5]  # the empty row is dropped
    assert bars[0]["high"] == 11.0
    assert bars[1]["volume"] == 120.0


@pytest.mark.unit
def test_a_csv_without_prices_is_an_error_not_an_empty_chart(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,signal\n2026-01-01,buy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no close column"):
        replay_bars_from_csv(path)


@pytest.mark.unit
def test_replay_bars_must_match_the_replay_feeds(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text("date,close\n2026-01-01,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a replay feed"):
        replay_bars_for(["SYNTH"], {"OTHER": str(path)})
    with pytest.raises(ValueError, match="need bars"):
        replay_bars_for(["SYNTH"], {})
    assert "SYNTH" in replay_bars_for(["SYNTH"], {"SYNTH": str(path)})


@pytest.mark.unit
def test_the_websocket_error_type_is_the_modules_own():
    # A URL typo should not surface as a bare ValueError from urlparse.
    with pytest.raises(WebSocketError):
        WebSocket("ftp://example.invalid")
    assert base64.b64encode(b"abc").decode()  # base64 is used for the nonce


# --------------------------------------------------------------------------------------
# The command line's feed builder
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_serve_command_builds_a_hub_only_when_asked():
    import typer

    from cli.research import _feed_hub

    assert _feed_hub(None, None, None, None) is None
    with pytest.raises(typer.Exit):
        _feed_hub(None, ["wss://ws.kite.trade"], None, None)

    hub = _feed_hub(["binance:BTCUSDT"], None, None, None)
    assert hub is not None
    assert [spec.symbol for spec in hub.specs()] == ["BTCUSDT"]
    assert hub.specs()[0].resolved_url().endswith("btcusdt@trade")


@pytest.mark.unit
def test_the_serve_command_requires_bars_for_a_replay_feed(tmp_path):
    import typer

    from cli.research import _feed_hub

    with pytest.raises(typer.Exit):
        _feed_hub(["replay:SYNTH"], None, None, None)
    with pytest.raises(typer.Exit):
        _feed_hub(["replay:SYNTH"], None, None, ["SYNTH"])  # no =PATH

    path = tmp_path / "bars.csv"
    path.write_text("date,close\n2026-01-01,10\n2026-01-02,11\n", encoding="utf-8")
    hub = _feed_hub(["replay:SYNTH"], None, None, [f"SYNTH={path}"])
    assert hub is not None and hub.specs()[0].provider == "replay"


@pytest.mark.unit
def test_the_serve_command_refuses_a_broker_feed_without_an_endpoint():
    import typer

    from cli.research import _feed_hub

    with pytest.raises(typer.Exit):
        _feed_hub(["generic:RELIANCE"], None, None, None)
    hub = _feed_hub(["generic:RELIANCE"], ["wss://ws.kite.trade"], ['{"a":"subscribe"}'], None)
    assert hub is not None
    assert hub.specs()[0].resolved_url() == "wss://ws.kite.trade"


@pytest.mark.unit
def test_a_feed_that_talks_but_never_prices_says_so():
    """A connected socket emitting unrecognised messages is a bug worth naming."""
    chatter = [json.dumps({"type": "heartbeat", "n": index})
               for index in range(feeds._SILENT_AFTER)]
    transport = FakeTransport(chatter)
    feed = WebSocketFeed(FeedSpec(symbol="RELIANCE", provider="generic",
                                  url="wss://broker.invalid/ws", subscribe="{}"),
                         BarSeries("RELIANCE"), transport_factory=lambda url: transport,
                         reconnect=False)
    feed.run()
    # The peer closing afterwards is expected; what matters is what the feed recorded
    # while it was still connected.
    state = feed.state()
    assert "recognisable price" in feed.error
    assert "subscribe payload" in feed.error
    assert state["quality"] == LIVE


@pytest.mark.unit
def test_the_warning_is_withdrawn_once_a_price_arrives():
    chatter = [json.dumps({"type": "heartbeat"}) for _ in range(feeds._SILENT_AFTER - 1)]
    chatter.append(json.dumps({"last_price": 2875.4, "last_traded_quantity": 5}))
    transport = FakeTransport(chatter)
    feed = WebSocketFeed(FeedSpec(symbol="RELIANCE", provider="generic",
                                  url="wss://broker.invalid/ws", subscribe="{}"),
                         BarSeries("RELIANCE"), transport_factory=lambda url: transport,
                         reconnect=False)
    feed.run()
    assert feed.series.last_price == 2875.4
    assert "recognisable price" not in feed.error
