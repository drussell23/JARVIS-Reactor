"""reactor-core inbound cross-repo ripple listener — independent-verification tests.

Simulates a JARVIS-signed ripple via the BYTE-IDENTICAL vendored contract,
then drives the listener: a valid ripple verifies and emits ONE local intent;
forged / replayed / expired / wrong-origin / wrong-PSK ripples are silently
DROPPED with no intent; the listener never executes the payload.
"""

from __future__ import annotations

import time

import pytest

from cross_repo_mesh.ripple_contract import (
    RipplePayload,
    VerifyVerdict,
    sign_ripple,
)
from cross_repo_mesh import ripple_listener as L

_PSK = b"shared-cross-repo-secret-32-bytes!!"
_WRONG_PSK = b"a-totally-different-secret-value!!!!"


def _payload(*, now, intent="capability graduated: semantic_index v1", nonce="n-aaa",
             origin="jarvis", ttl=3600.0):
    return RipplePayload(
        schema_version="cross_repo_ripple.1",
        ripple_kind="capability_graduated",
        source_repo=origin,
        intent=intent,
        payload_sha256="0" * 64,
        nonce=nonce,
        issued_at_unix=now,
        ttl_s=ttl,
    )


@pytest.fixture(autouse=True)
def _enable(monkeypatch, tmp_path):
    monkeypatch.setenv("REACTOR_CORE_RIPPLE_LISTENER_ENABLED", "true")
    monkeypatch.setenv("JARVIS_CROSS_REPO_EMIT_PSK", _PSK.decode())
    monkeypatch.setenv(
        "REACTOR_CORE_RIPPLE_INTENT_LEDGER_PATH",
        str(tmp_path / "inbound_intents.jsonl"),
    )
    yield


def test_valid_ripple_verifies_and_emits_local_intent():
    now = time.time()
    token = sign_ripple(_payload(now=now), _PSK)
    verdict, intent = L.handle_inbound_ripple(token, now_unix=now, seen_nonces=set())
    assert verdict is VerifyVerdict.VERIFIED
    assert intent is not None
    assert intent["received_by"] == "reactor-core"
    assert intent["ripple_kind"] == "capability_graduated"
    assert intent["origin"] == "jarvis"


def test_tampered_payload_dropped_no_intent():
    now = time.time()
    token = sign_ripple(_payload(now=now), _PSK)
    head, sig = token.split(".", 1)
    bad = head[:-1] + ("A" if head[-1] != "A" else "B") + "." + sig
    verdict, intent = L.handle_inbound_ripple(bad, now_unix=now, seen_nonces=set())
    assert verdict is VerifyVerdict.DROPPED_BAD_SIGNATURE
    assert intent is None


def test_wrong_psk_dropped():
    now = time.time()
    token = sign_ripple(_payload(now=now), _WRONG_PSK)
    verdict, intent = L.handle_inbound_ripple(token, now_unix=now, seen_nonces=set())
    assert verdict is VerifyVerdict.DROPPED_BAD_SIGNATURE
    assert intent is None


def test_replay_dropped_second_time():
    now = time.time()
    token = sign_ripple(_payload(now=now, nonce="n-replay"), _PSK)
    seen = set()
    v1, _ = L.handle_inbound_ripple(token, now_unix=now, seen_nonces=seen)
    v2, i2 = L.handle_inbound_ripple(token, now_unix=now, seen_nonces=seen)
    assert v1 is VerifyVerdict.VERIFIED
    assert v2 is VerifyVerdict.DROPPED_REPLAY
    assert i2 is None


def test_expired_dropped():
    now = time.time()
    token = sign_ripple(_payload(now=now, ttl=10.0), _PSK)
    verdict, intent = L.handle_inbound_ripple(token, now_unix=now + 100, seen_nonces=set())
    assert verdict is VerifyVerdict.DROPPED_EXPIRED
    assert intent is None


def test_wrong_origin_dropped():
    now = time.time()
    token = sign_ripple(_payload(now=now, origin="evil-repo"), _PSK)
    verdict, intent = L.handle_inbound_ripple(token, now_unix=now, seen_nonces=set())
    assert verdict is VerifyVerdict.DROPPED_WRONG_ORIGIN
    assert intent is None


def test_master_off_is_disabled(monkeypatch):
    monkeypatch.delenv("REACTOR_CORE_RIPPLE_LISTENER_ENABLED", raising=False)
    now = time.time()
    token = sign_ripple(_payload(now=now), _PSK)
    verdict, intent = L.handle_inbound_ripple(token, now_unix=now, seen_nonces=set())
    assert verdict is VerifyVerdict.DISABLED
    assert intent is None


def test_no_psk_is_disabled_never_accepts(monkeypatch):
    monkeypatch.delenv("JARVIS_CROSS_REPO_EMIT_PSK", raising=False)
    now = time.time()
    token = sign_ripple(_payload(now=now), _PSK)
    verdict, intent = L.handle_inbound_ripple(token, now_unix=now, seen_nonces=set())
    assert verdict is VerifyVerdict.DISABLED
    assert intent is None


def test_dangerous_intent_string_is_never_executed(tmp_path):
    marker = tmp_path / "PWNED"
    now = time.time()
    evil = f"__import__('os').system('touch {marker}')"
    token = sign_ripple(_payload(now=now, intent=evil), _PSK)
    verdict, intent = L.handle_inbound_ripple(token, now_unix=now, seen_nonces=set())
    assert verdict is VerifyVerdict.VERIFIED
    assert intent["intent"] == evil
    assert not marker.exists()
