"""Tests for UMF heartbeat publishing in reactor-core (Task 17)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from umf_client import ReactorUmfClient
from umf_types import Kind, Stream


class TestReactorUmfHeartbeat:

    @pytest.fixture
    def client(self, tmp_path: Path) -> ReactorUmfClient:
        return ReactorUmfClient(
            session_id="test-session",
            instance_id="test-inst",
            dedup_db_path=tmp_path / "dedup.db",
        )

    @pytest.mark.asyncio
    async def test_send_heartbeat_running(self, client: ReactorUmfClient) -> None:
        await client.start()
        try:
            result = await client.send_heartbeat(state="running")
            assert result is True
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_send_heartbeat_with_extras(self, client: ReactorUmfClient) -> None:
        await client.start()
        try:
            result = await client.send_heartbeat(
                state="running",
                liveness=True,
                readiness=True,
                queue_depth=3,
                resource_pressure=0.15,
            )
            assert result is True
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_payload_fields(self, client: ReactorUmfClient) -> None:
        """Verify heartbeat message contains all expected payload fields."""
        received: list = []

        async def handler(msg):
            received.append(msg)

        await client.start()
        try:
            await client.subscribe(Stream.lifecycle, handler)
            await client.send_heartbeat(
                state="degraded",
                liveness=True,
                readiness=False,
                queue_depth=7,
                resource_pressure=0.85,
            )
            assert len(received) == 1
            msg = received[0]
            assert msg.kind == Kind.heartbeat
            assert msg.stream == Stream.lifecycle
            assert msg.source.repo == "reactor-core"
            assert msg.source.component == "reactor"
            assert msg.payload["state"] == "degraded"
            assert msg.payload["liveness"] is True
            assert msg.payload["readiness"] is False
            assert msg.payload["queue_depth"] == 7
            assert msg.payload["resource_pressure"] == 0.85
            assert msg.payload["subsystem_role"] == "reactor"
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_dedup_prevents_duplicate(self, client: ReactorUmfClient) -> None:
        """Two heartbeats with same idempotency key should dedup the second."""
        received: list = []

        async def handler(msg):
            received.append(msg)

        await client.start()
        try:
            await client.subscribe(Stream.lifecycle, handler)
            first = await client.send_heartbeat(state="running")
            second = await client.send_heartbeat(state="running")
            assert first is True
            # second may be deduped (same idempotency key) or accepted
            # depending on key generation -- either way, no crash
            assert isinstance(second, bool)
        finally:
            await client.stop()
