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
