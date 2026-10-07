#!/usr/bin/env bash
# Start, stop or query the Reactor-Core API server (the Trinity's Nerves).
#
#   reactor_core.sh start | stop | status
#
# Idempotent: `start` returns at once if a server already answers /health,
# otherwise launches uvicorn detached (setsid + nohup, so it outlives the
# caller -- e.g. `ov`'s sibling bring-up) and waits for /health.
#
# Binds loopback: the server's own default host is 0.0.0.0, which under WSL
# mirrored networking exposes the training API to the LAN.
#
# Environment (all optional):
#   REACTOR_PYTHON       interpreter with reactor's [server] extra + ML stack
#                        (default ~/.venvs/reactor-train/bin/python)
#   REACTOR_PORT         8090
#   TRINITY_EVENTS_DIR   where O+V's trajectory recorder writes experience
#                        events (default ~/.jarvis/trinity/events). Without
#                        this the receiver watches an XDG default nobody
#                        writes to and ingests nothing.
#   REACTOR_WAIT_S       seconds to wait for /health on start (90)
set -u
ACTION="${1:-status}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REACTOR_PYTHON:-$HOME/.venvs/reactor-train/bin/python}"
PORT="${REACTOR_PORT:-8090}"
RUN="$HOME/.jarvis/run"; LOGS="$HOME/.jarvis/logs"
PIDFILE="$RUN/reactor-core-$PORT.pid"; LOG="$LOGS/reactor-core.log"
URL="http://127.0.0.1:$PORT/health"
mkdir -p "$RUN" "$LOGS"

serving() { curl -s -m 3 -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null | grep -q '^200$'; }
recorded_pid() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && cat "$PIDFILE"; }

case "$ACTION" in
  status)
    if serving; then echo "serving on :$PORT pid=$(recorded_pid || echo '?')"; exit 0; fi
    echo "not serving on :$PORT"; exit 3 ;;
  stop)
    pid="$(recorded_pid || true)"
    if [ -n "$pid" ]; then kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"; rm -f "$PIDFILE"; echo "stopped pid $pid"
    else echo "no reactor-core recorded on :$PORT"; fi
    exit 0 ;;
  start)
    if serving; then echo "already serving on :$PORT"; exit 0; fi
    [ -x "$PY" ] || { echo "reactor python not found: $PY"; exit 1; }
    export TRINITY_EVENTS_DIR="${TRINITY_EVENTS_DIR:-$HOME/.jarvis/trinity/events}"
    export REACTOR_CORE_HOST=127.0.0.1 REACTOR_PORT="$PORT"
    cd "$REPO" || exit 1
    # The server's own main(): it configures reactor_core logging (a bare
    # `uvicorn module:app` leaves it unconfigured and every INFO line --
    # ingestion, flushes, refusals -- is silently dropped) and binds
    # REACTOR_CORE_HOST / REACTOR_PORT exported above.
    setsid nohup "$PY" -m reactor_core.api.server >>"$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    deadline=$(( $(date +%s) + ${REACTOR_WAIT_S:-90} ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "reactor-core exited; see $LOG"; exit 1; }
      if serving; then echo "started pid $(cat "$PIDFILE") on :$PORT (events: $TRINITY_EVENTS_DIR)"; exit 0; fi
      sleep 1
    done
    echo "reactor-core did not answer within ${REACTOR_WAIT_S:-90}s; see $LOG"; exit 1 ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
