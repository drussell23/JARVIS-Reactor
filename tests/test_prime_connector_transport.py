"""Tests for PrimeConnector transport negotiation and fallback."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from reactor_core.integration.prime_connector import (
    ConnectionState,
    PrimeConnector,
    PrimeConnectorConfig,
    PrimeEvent,
    PrimeEventType,
)


def test_prime_event_from_dict_accepts_prime_websocket_envelope():
    event = PrimeEvent.from_dict(
        {
            "event_type": "heartbeat",
            "timestamp": datetime.now().isoformat(),
            "data": {"status": "ready"},
        }
    )
    assert event.event_type == PrimeEventType.HEALTH
    assert event.success is True
    assert event.metadata["payload"]["status"] == "ready"


@pytest.mark.asyncio
async def test_connect_websocket_rotates_paths_on_contract_error(monkeypatch):
    config = PrimeConnectorConfig(
        host="localhost",
        port=8002,
        enable_websocket=True,
        websocket_path="/ws/events",
        websocket_paths=["/ws/events", "/ws/alt"],
        max_reconnect_attempts=1,
    )
    connector = PrimeConnector(config)

    class FakeSession:
        def __init__(self):
            self.calls = []

        async def ws_connect(self, url, heartbeat=None):
            self.calls.append((url, heartbeat))
            if url.endswith("/ws/events"):
                raise RuntimeError("HTTP 404")
            return object()

    fake_session = FakeSession()

    async def fake_get_session():
        return fake_session

    async def fake_listener():
        return None

    monkeypatch.setattr(connector, "_get_session", fake_get_session)
    monkeypatch.setattr(connector, "_ws_listener", fake_listener)

    await connector.connect_websocket()

    assert connector._ws_state == ConnectionState.CONNECTED
    assert fake_session.calls[0][0].endswith("/ws/events")
    assert fake_session.calls[1][0].endswith("/ws/alt")
    assert connector.config.websocket_path == "/ws/alt"


@pytest.mark.asyncio
async def test_stream_events_uses_health_fallback_when_websocket_disabled(monkeypatch):
    config = PrimeConnectorConfig(
        enable_websocket=False,
        enable_health_poll_fallback=True,
        health_poll_interval=0.01,
    )
    connector = PrimeConnector(config)

    async def fake_health_event():
        return PrimeEvent(
            event_id="health_1",
            event_type=PrimeEventType.HEALTH,
            timestamp=datetime.now(),
            success=True,
            confidence=1.0,
            metadata={"health": {"status": "healthy"}},
        )

    monkeypatch.setattr(connector, "_build_health_event", fake_health_event)

    stream = connector.stream_events()
    event = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
    assert event.event_type == PrimeEventType.HEALTH
    await stream.aclose()


class _FakeHTTPResponse:
    def __init__(self, status: int, payload):
        self.status = status
        self._payload = payload

    async def json(self, content_type=None):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_discover_ws_contract_updates_paths_and_heartbeat(monkeypatch):
    config = PrimeConnectorConfig(
        host="localhost",
        port=8002,
        websocket_paths=["/ws/events"],
        health_paths=["/health"],
        contract_paths=["/api/v1/integration/contracts/ws-events"],
    )
    connector = PrimeConnector(config)

    contract_payload = {
        "contract_type": "jarvis_prime.ws_events",
        "contract_version": "1.0",
        "websocket": {
            "primary_path": "/ws/contract-events",
            "paths": ["/ws/contract-events", "/ws/events"],
            "heartbeat_seconds": 12.0,
        },
        "health": {"paths": ["/health", "/readyz"]},
    }

    class FakeSession:
        def __init__(self):
            self.get_calls = []

        def get(self, url):
            self.get_calls.append(url)
            if url.endswith("/api/v1/integration/contracts/ws-events"):
                return _FakeHTTPResponse(200, contract_payload)
            return _FakeHTTPResponse(404, {"error": "not found"})

    fake_session = FakeSession()

    async def fake_get_session():
        return fake_session

    monkeypatch.setattr(connector, "_get_session", fake_get_session)

    discovered = await connector.discover_ws_contract(force=True)

    assert discovered is not None
    assert connector.config.websocket_path == "/ws/contract-events"
    assert connector.config.websocket_paths[0] == "/ws/contract-events"
    assert connector.config.websocket_paths[1] == "/ws/events"
    assert connector.config.health_paths[0] == "/health"
    assert connector.config.health_paths[1] == "/readyz"
    assert connector.config.ping_interval == 12.0


@pytest.mark.asyncio
async def test_connect_websocket_uses_discovered_contract_path(monkeypatch):
    config = PrimeConnectorConfig(
        host="localhost",
        port=8002,
        enable_websocket=True,
        websocket_path="/ws/events",
        websocket_paths=["/ws/events"],
        contract_paths=["/api/v1/integration/contracts/ws-events"],
    )
    connector = PrimeConnector(config)

    class FakeSession:
        def __init__(self):
            self.get_calls = []
            self.ws_calls = []

        def get(self, url):
            self.get_calls.append(url)
            if url.endswith("/api/v1/integration/contracts/ws-events"):
                return _FakeHTTPResponse(
                    200,
                    {
                        "websocket": {
                            "primary_path": "/ws/contract-events",
                            "paths": ["/ws/contract-events", "/ws/events"],
                            "heartbeat_seconds": 15.0,
                        }
                    },
                )
            return _FakeHTTPResponse(404, {"error": "not found"})

        async def ws_connect(self, url, heartbeat=None):
            self.ws_calls.append((url, heartbeat))
            if not url.endswith("/ws/contract-events"):
                raise RuntimeError("HTTP 404")
            return object()

    fake_session = FakeSession()

    async def fake_get_session():
        return fake_session

    async def fake_listener():
        return None

    monkeypatch.setattr(connector, "_get_session", fake_get_session)
    monkeypatch.setattr(connector, "_ws_listener", fake_listener)

    await connector.connect_websocket()

    assert connector._ws_state == ConnectionState.CONNECTED
    assert fake_session.ws_calls[0][0].endswith("/ws/contract-events")
    assert fake_session.ws_calls[0][1] == 15.0


@pytest.mark.asyncio
async def test_connect_websocket_refreshes_contract_after_endpoint_mismatch(monkeypatch):
    config = PrimeConnectorConfig(
        host="localhost",
        port=8002,
        enable_websocket=True,
        websocket_path="/ws/events",
        websocket_paths=["/ws/events"],
        health_paths=["/health"],
        contract_paths=["/api/v1/integration/contracts/ws-events"],
    )
    connector = PrimeConnector(config)

    class FakeSession:
        def __init__(self):
            self.get_calls = []
            self.ws_calls = []
            self.contract_calls = 0

        def get(self, url):
            self.get_calls.append(url)
            if url.endswith("/api/v1/integration/contracts/ws-events"):
                self.contract_calls += 1
                if self.contract_calls == 1:
                    return _FakeHTTPResponse(404, {"error": "not found"})
                return _FakeHTTPResponse(
                    200,
                    {
                        "websocket": {
                            "primary_path": "/ws/contract-events",
                            "paths": ["/ws/contract-events", "/ws/events"],
                            "heartbeat_seconds": 9.0,
                        }
                    },
                )
            if url.endswith("/health"):
                return _FakeHTTPResponse(200, {"status": "starting"})
            return _FakeHTTPResponse(404, {"error": "not found"})

        async def ws_connect(self, url, heartbeat=None):
            self.ws_calls.append((url, heartbeat))
            if url.endswith("/ws/contract-events"):
                return object()
            raise RuntimeError("HTTP 404")

    fake_session = FakeSession()

    async def fake_get_session():
        return fake_session

    async def fake_listener():
        return None

    monkeypatch.setattr(connector, "_get_session", fake_get_session)
    monkeypatch.setattr(connector, "_ws_listener", fake_listener)

    await connector.connect_websocket()

    assert connector._ws_state == ConnectionState.CONNECTED
    assert connector.config.websocket_path == "/ws/contract-events"
    assert fake_session.contract_calls >= 2
    assert any(call.endswith("/health") for call in fake_session.get_calls)
    assert fake_session.ws_calls[-1][0].endswith("/ws/contract-events")
    assert fake_session.ws_calls[-1][1] == 9.0
