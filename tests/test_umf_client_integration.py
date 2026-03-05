"""Integration tests for ReactorUmfClient -- heartbeat and event publishing."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from umf_client import ReactorUmfClient
from umf_types import Kind, Stream


class TestReactorUmfClient:
    """Verify ReactorUmfClient can send heartbeats and publish events."""

    @pytest.mark.asyncio
    async def test_reactor_can_send_heartbeat(self, tmp_path: Path) -> None:
        """Create client, subscribe to lifecycle, send heartbeat, verify source.repo."""
        received: list = []

        async def handler(msg):
            received.append(msg)

        client = ReactorUmfClient(
            session_id="sess-001",
            instance_id="inst-001",
            dedup_db_path=tmp_path / "dedup.db",
        )
        await client.start()
        try:
            await client.subscribe(Stream.lifecycle, handler)
            ok = await client.send_heartbeat(state="ready")
            assert ok is True
            assert len(received) == 1
            msg = received[0]
            assert msg.source.repo == "reactor-core"
            assert msg.stream == Stream.lifecycle
            assert msg.kind == Kind.heartbeat
            assert msg.payload["state"] == "ready"
            assert msg.payload["liveness"] is True
            assert msg.payload["readiness"] is True
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_reactor_can_publish_event(self, tmp_path: Path) -> None:
        """Create client, subscribe to event, publish_event, verify kind==Kind.event."""
        received: list = []

        async def handler(msg):
            received.append(msg)

        client = ReactorUmfClient(
            session_id="sess-002",
            instance_id="inst-002",
            dedup_db_path=tmp_path / "dedup.db",
        )
        await client.start()
        try:
            await client.subscribe(Stream.event, handler)
            ok = await client.publish_event(
                target_repo="jarvis",
                target_component="supervisor",
                payload={"action": "training_complete", "model": "v3"},
            )
            assert ok is True
            assert len(received) == 1
            msg = received[0]
            assert msg.kind == Kind.event
            assert msg.stream == Stream.event
            assert msg.source.repo == "reactor-core"
            assert msg.source.component == "reactor"
            assert msg.target.repo == "jarvis"
            assert msg.target.component == "supervisor"
            assert msg.payload["action"] == "training_complete"
        finally:
            await client.stop()
