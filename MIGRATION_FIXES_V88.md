# Trinity System Migration to v88.0 - Specific Code Fixes

This document contains **exact code changes** needed to fix the CancelledError cascades across the Trinity system.

---

## 🎯 File 1: `/JARVIS-AI-Agent/backend/neural_mesh/agents/visual_monitor_agent.py`

### Location: Line 1256-1282 (`init_detector` function)

**Current Code (BROKEN):**
```python
async def init_detector():
    """OCR Detector - Non-blocking (can be slow due to model loading)."""
    comp_start = time_module.time()
    try:
        def _init_detector():
            from backend.vision.visual_event_detector import create_detector
            return create_detector()

        # v89.0: Use asyncio.to_thread() for proper event loop handling
        self._detector = await asyncio.wait_for(
            asyncio.to_thread(_init_detector),
            timeout=detector_init_timeout
        )
        component_status["detector"]["success"] = True
        component_status["detector"]["duration"] = time_module.time() - comp_start
        logger.info("✅ OCR Detector Ready")

    except asyncio.TimeoutError:
        component_status["detector"]["error"] = f"timeout ({detector_init_timeout}s)"
        component_status["detector"]["duration"] = time_module.time() - comp_start
        logger.warning(f"VisualEventDetector timeout after {detector_init_timeout}s")
        self._detector = None
    except Exception as e:
        component_status["detector"]["error"] = str(e)
        component_status["detector"]["duration"] = time_module.time() - comp_start
        logger.warning(f"VisualEventDetector failed: {e}")
        self._detector = None
```

**Fixed Code (v88.0):**
```python
# Add this import at the top of the file
from reactor_core.integration import wait_for_safe

async def init_detector():
    """OCR Detector - Cancellation-safe, non-blocking."""
    comp_start = time_module.time()

    def _init_detector():
        from backend.vision.visual_event_detector import create_detector
        return create_detector()

    async def _cleanup_detector():
        """Cleanup partial initialization if cancelled."""
        logger.debug("Cleaning up detector initialization")
        if hasattr(self, '_partial_detector'):
            try:
                await self._partial_detector.close()
                del self._partial_detector
            except:
                pass

    # v88.0: Use wait_for_safe for cancellation-safe timeout handling
    self._detector = await wait_for_safe(
        asyncio.to_thread(_init_detector),
        timeout=detector_init_timeout,
        cleanup=_cleanup_detector,
        default=None,
    )

    # Update status
    component_status["detector"]["duration"] = time_module.time() - comp_start

    if self._detector is not None:
        component_status["detector"]["success"] = True
        logger.info("✅ OCR Detector Ready")
    else:
        component_status["detector"]["error"] = "timeout or cancelled"
        logger.warning(f"⚠️ Detector initialization failed/timeout/cancelled")
```

**Why This Works:**
- ✅ Handles `TimeoutError` automatically
- ✅ Handles `CancelledError` without cascade
- ✅ Cleanup runs even if cancelled
- ✅ Returns `None` on any failure (timeout/cancel/error)
- ✅ No nested exception handling

---

## 🎯 File 2: `/JARVIS-AI-Agent/backend/neural_mesh/agents/visual_monitor_agent.py`

### Location: Similar pattern for all component initialization functions

Apply the same fix to:
- `init_computer_use()` (if similar pattern exists)
- `init_screen_monitor()` (if similar pattern exists)
- Any other component initialization functions

**Pattern to Fix:**
```python
# ❌ BEFORE - Any function using bare asyncio.wait_for
try:
    result = await asyncio.wait_for(some_coro(), timeout=X)
except asyncio.TimeoutError:
    # handle timeout
except Exception as e:
    # handle error
# Missing: CancelledError handling!
```

```python
# ✅ AFTER - Use wait_for_safe
from reactor_core.integration import wait_for_safe

result = await wait_for_safe(
    some_coro(),
    timeout=X,
    cleanup=optional_cleanup_function,
    default=default_value_on_failure,
)
# Handles ALL cases: timeout, cancel, error
```

---

## 🎯 File 3: `/JARVIS-AI-Agent/backend/main.py` (or similar startup file)

### Add Lifecycle Coordinator Integration

**Add at the top of file:**
```python
from reactor_core.integration import (
    get_lifecycle_coordinator,
    TaskPriority,
    cancellation_safe_decorator,
)
```

**Wrap main async function:**
```python
# ❌ BEFORE
async def start_backend():
    """Start JARVIS backend."""
    # Initialize components
    await initialize_neural_mesh()
    await start_voice_system()
    await start_visual_monitor()

    # Run main loop
    while True:
        await process_tasks()
        await asyncio.sleep(0.1)
```

```python
# ✅ AFTER
from reactor_core.integration import get_lifecycle_coordinator, TaskPriority

async def _cleanup_backend():
    """Cleanup backend resources."""
    logger.info("Cleaning up JARVIS backend")
    await cleanup_neural_mesh()
    await cleanup_voice_system()
    await cleanup_visual_monitor()

@cancellation_safe_decorator(cleanup_attr="_cleanup_backend")
async def start_backend():
    """Start JARVIS backend - cancellation-safe."""
    # Get lifecycle coordinator
    coordinator = await get_lifecycle_coordinator()

    # Create managed tasks
    neural_mesh_task = await coordinator.create_task(
        initialize_neural_mesh(),
        name="neural_mesh",
        priority=TaskPriority.HIGH,
    )

    voice_task = await coordinator.create_task(
        start_voice_system(),
        name="voice_system",
        priority=TaskPriority.HIGH,
    )

    visual_task = await coordinator.create_task(
        start_visual_monitor(),
        name="visual_monitor",
        priority=TaskPriority.NORMAL,
    )

    # Wait for initialization
    await asyncio.gather(neural_mesh_task, voice_task, visual_task)

    # Run main loop (will be cancelled gracefully)
    while True:
        await process_tasks()
        await asyncio.sleep(0.1)
```

---

## 🎯 File 4: `/run_supervisor.py` (Trinity Unified Startup)

### Complete Rewrite for v88.0

**New `/run_supervisor.py` (Trinity v88.0):**
```python
#!/usr/bin/env python3
"""
Trinity Unified Supervisor (v88.0)
Single command to start JARVIS Body + Prime + Reactor Core

Usage:
    python3 run_supervisor.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add repos to path
sys.path.insert(0, str(Path(__file__).parent / "JARVIS-AI-Agent"))
sys.path.insert(0, str(Path(__file__).parent / "jarvis-prime"))
sys.path.insert(0, str(Path(__file__).parent / "reactor-core"))

from reactor_core.integration import (
    get_lifecycle_coordinator,
    get_unified_coordinator,
    get_health_monitor,
    get_db_coordinator,
    TaskPriority,
    ShutdownPhase,
    ShutdownConfig,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# COMPONENT STARTUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


async def start_jarvis_body():
    """Start JARVIS Body (AI Agent with neural mesh)."""
    logger.info("🤖 Starting JARVIS Body...")
    from backend.main import start_backend
    await start_backend()


async def start_jarvis_prime():
    """Start JARVIS Prime (WebSocket/REST server)."""
    logger.info("🔷 Starting JARVIS Prime...")
    from jarvis_prime.server import start_server
    await start_server()


async def start_reactor_core():
    """Start Reactor Core (Training pipeline)."""
    logger.info("⚛️ Starting Reactor Core...")
    from reactor_core.orchestration import start_training_pipeline
    await start_training_pipeline()


# ═══════════════════════════════════════════════════════════════════
# COMPONENT CLEANUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


async def cleanup_jarvis_body():
    """Cleanup JARVIS Body resources."""
    logger.info("Cleaning up JARVIS Body...")
    try:
        from backend.main import cleanup_backend
        await cleanup_backend()
    except Exception as e:
        logger.error(f"JARVIS Body cleanup failed: {e}")


async def cleanup_jarvis_prime():
    """Cleanup JARVIS Prime resources."""
    logger.info("Cleaning up JARVIS Prime...")
    try:
        from jarvis_prime.server import cleanup_server
        await cleanup_server()
    except Exception as e:
        logger.error(f"JARVIS Prime cleanup failed: {e}")


async def cleanup_reactor_core():
    """Cleanup Reactor Core resources."""
    logger.info("Cleaning up Reactor Core...")
    try:
        from reactor_core.orchestration import cleanup_training
        await cleanup_training()
    except Exception as e:
        logger.error(f"Reactor Core cleanup failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# MAIN TRINITY ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════


async def main():
    """
    Trinity Unified Startup - v88.0 with bulletproof lifecycle management.
    """
    print("━" * 80)
    print("  JARVIS TRINITY - UNIFIED SYSTEM STARTUP (v88.0)")
    print("━" * 80)
    print()

    # Configure shutdown behavior
    shutdown_config = ShutdownConfig(
        graceful_timeout=30.0,  # 30s for graceful shutdown
        forceful_timeout=10.0,  # 10s for forceful shutdown
        terminate_timeout=5.0,  # 5s before giving up
        shield_cleanup=True,  # Always shield cleanup
        log_orphaned_tasks=True,  # Log stubborn tasks
    )

    # Get lifecycle coordinator (singleton)
    lifecycle = await get_lifecycle_coordinator(shutdown_config)

    # Install signal handlers (Ctrl+C, SIGTERM)
    lifecycle.install_signal_handlers()
    logger.info("✅ Signal handlers installed (Ctrl+C for graceful shutdown)")

    try:
        # ═════════════════════════════════════════════════════════════
        # PHASE 1: Initialize Trinity Infrastructure
        # ═════════════════════════════════════════════════════════════

        logger.info("Phase 1: Initializing Trinity infrastructure...")

        # Start coordinators (CRITICAL priority - shutdown last)
        unified_coord = await get_unified_coordinator()
        health_monitor = await get_health_monitor()
        db_coordinator = await get_db_coordinator()

        # Create infrastructure tasks
        await lifecycle.create_task(
            unified_coord.run(),
            name="unified_coordinator",
            priority=TaskPriority.CRITICAL,
        )

        await lifecycle.create_task(
            health_monitor.start(),
            name="health_monitor",
            priority=TaskPriority.CRITICAL,
        )

        logger.info("✅ Trinity infrastructure ready")

        # ═════════════════════════════════════════════════════════════
        # PHASE 2: Start Trinity Components
        # ═════════════════════════════════════════════════════════════

        logger.info("Phase 2: Starting Trinity components...")

        # JARVIS Body (HIGH priority)
        jarvis_task = await lifecycle.create_task(
            start_jarvis_body(),
            name="jarvis_body",
            priority=TaskPriority.HIGH,
            cleanup=cleanup_jarvis_body,
        )

        # JARVIS Prime (HIGH priority)
        prime_task = await lifecycle.create_task(
            start_jarvis_prime(),
            name="jarvis_prime",
            priority=TaskPriority.HIGH,
            cleanup=cleanup_jarvis_prime,
        )

        # Reactor Core (NORMAL priority)
        reactor_task = await lifecycle.create_task(
            start_reactor_core(),
            name="reactor_core",
            priority=TaskPriority.NORMAL,
            cleanup=cleanup_reactor_core,
        )

        logger.info("✅ All Trinity components started")

        # ═════════════════════════════════════════════════════════════
        # PHASE 3: Run Until Shutdown Signal
        # ═════════════════════════════════════════════════════════════

        print()
        print("━" * 80)
        print("  ✅ TRINITY SYSTEM FULLY OPERATIONAL")
        print("  📊 Components: JARVIS Body | JARVIS Prime | Reactor Core")
        print("  🛑 Press Ctrl+C for graceful shutdown")
        print("━" * 80)
        print()

        # Wait for all main tasks (this blocks until shutdown signal)
        await asyncio.gather(
            jarvis_task,
            prime_task,
            reactor_task,
            return_exceptions=True,
        )

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")

    except asyncio.CancelledError:
        logger.info("Main task cancelled")

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)

    finally:
        # ═════════════════════════════════════════════════════════════
        # PHASE 4: Graceful Shutdown
        # ═════════════════════════════════════════════════════════════

        print()
        print("━" * 80)
        print("  🛑 INITIATING GRACEFUL SHUTDOWN")
        print("━" * 80)

        logger.info("Shutting down Trinity system...")

        # Execute coordinated shutdown
        report = await lifecycle.shutdown(phase=ShutdownPhase.GRACEFUL)

        # Print shutdown report
        print()
        print("📊 Shutdown Report:")
        print(f"  Total tasks: {report.get('total', 0)}")
        print(f"  Cancelled: {report.get('cancelled', 0)}")
        print(f"  Failed: {report.get('failed', 0)}")
        print(f"  Duration: {report.get('duration', 0):.1f}s")
        print()

        if report.get('stubborn_tasks'):
            print("⚠️  Stubborn tasks (refused to cancel):")
            for task in report['stubborn_tasks']:
                print(f"    - {task['name']} (runtime: {task['runtime']:.1f}s)")
            print()

        print("━" * 80)
        print("  ✅ TRINITY SHUTDOWN COMPLETE")
        print("━" * 80)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        # Run Trinity system
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n✅ Shutdown complete")

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
```

---

## 🎯 File 5: Add Cleanup Functions to Each Repo

### `/JARVIS-AI-Agent/backend/main.py`

**Add cleanup function:**
```python
async def cleanup_backend():
    """Cleanup JARVIS backend resources."""
    logger.info("Cleaning up JARVIS backend...")

    # Stop all neural mesh agents
    if hasattr(global_state, 'neural_mesh'):
        await global_state.neural_mesh.stop()

    # Stop voice system
    if hasattr(global_state, 'voice_system'):
        await global_state.voice_system.stop()

    # Close database connections
    if hasattr(global_state, 'db'):
        await global_state.db.close()

    logger.info("✅ JARVIS backend cleanup complete")
```

### `/jarvis-prime/server.py`

**Add cleanup function:**
```python
async def cleanup_server():
    """Cleanup JARVIS Prime server resources."""
    logger.info("Cleaning up JARVIS Prime...")

    # Close WebSocket connections
    if hasattr(app_state, 'websocket_manager'):
        await app_state.websocket_manager.close_all()

    # Stop HTTP server
    if hasattr(app_state, 'http_server'):
        await app_state.http_server.close()

    logger.info("✅ JARVIS Prime cleanup complete")
```

### `/reactor-core/reactor_core/orchestration/__init__.py`

**Add cleanup function:**
```python
async def cleanup_training():
    """Cleanup Reactor Core training resources."""
    logger.info("Cleaning up Reactor Core...")

    # Stop training loops
    if hasattr(training_state, 'active_trainers'):
        for trainer in training_state.active_trainers:
            await trainer.stop()

    # Save checkpoints
    if hasattr(training_state, 'checkpoint_manager'):
        await training_state.checkpoint_manager.save_final_checkpoint()

    logger.info("✅ Reactor Core cleanup complete")
```

---

## 🎯 File 6: Update Imports Across All Files

### Add to files that use asyncio operations:

```python
# Add these imports to any file using asyncio.wait_for, gather, etc.
from reactor_core.integration import (
    wait_for_safe,           # Replace asyncio.wait_for
    gather_safe,             # Replace asyncio.gather
    cancellation_safe,       # Wrap critical operations
    cancellation_safe_decorator,  # Decorate async methods
)
```

### Search and Replace Patterns:

1. **Find:** `await asyncio.wait_for(`
   **Replace with:** `await wait_for_safe(`
   (Also add `cleanup=` and `default=` parameters)

2. **Find:** `await asyncio.gather(`
   **Replace with:** `await gather_safe(`
   (Also add `cleanup=` parameter if needed)

3. **Find patterns like:**
   ```python
   try:
       # operation
   except asyncio.CancelledError:
       # cleanup
       raise
   ```
   **Replace with:**
   ```python
   await cancellation_safe(
       operation(),
       cleanup=cleanup_function,
   )
   ```

---

## 🧪 Testing the Migration

### Test 1: Graceful Shutdown

```bash
# Start Trinity system
python3 run_supervisor.py

# Wait for "TRINITY SYSTEM FULLY OPERATIONAL"

# Press Ctrl+C
# Expected output:
#   🛑 INITIATING GRACEFUL SHUTDOWN
#   Shutting down Trinity system...
#   📊 Shutdown Report:
#     Total tasks: 5
#     Cancelled: 5
#     Failed: 0
#     Duration: 2.3s
#   ✅ TRINITY SHUTDOWN COMPLETE
```

### Test 2: No More CancelledError Cascades

```bash
# Monitor logs during shutdown
python3 run_supervisor.py 2>&1 | tee trinity.log

# Press Ctrl+C

# Check log for errors:
grep -i "CancelledError" trinity.log

# Expected: NO "During handling of the above exception" messages
```

### Test 3: All Components Cleanup

```bash
# Start system
python3 run_supervisor.py

# Press Ctrl+C

# Check logs for cleanup messages:
# Expected to see:
#   Cleaning up JARVIS Body...
#   Cleaning up JARVIS Prime...
#   Cleaning up Reactor Core...
#   ✅ JARVIS backend cleanup complete
#   ✅ JARVIS Prime cleanup complete
#   ✅ Reactor Core cleanup complete
```

---

## 📋 Migration Checklist

- [ ] Add `from reactor_core.integration import wait_for_safe` to visual_monitor_agent.py
- [ ] Replace `asyncio.wait_for` with `wait_for_safe` in init_detector()
- [ ] Add similar fixes to all component initialization functions
- [ ] Add cleanup functions: `cleanup_backend()`, `cleanup_server()`, `cleanup_training()`
- [ ] Update `run_supervisor.py` with v88.0 lifecycle coordinator
- [ ] Add `@cancellation_safe_decorator` to long-running async methods
- [ ] Test graceful shutdown with Ctrl+C
- [ ] Verify no "CancelledError during handling" errors in logs
- [ ] Verify all cleanup functions run during shutdown
- [ ] Test rapid Ctrl+C (should escalate to forceful shutdown)

---

## 🚨 Breaking Changes

**None!** The v88.0 async lifecycle coordinator is **backwards compatible**. You can:
1. Keep existing code running
2. Migrate incrementally, file by file
3. Mix old and new patterns

**Recommendation:** Prioritize fixing files with `asyncio.wait_for` first (highest risk for CancelledError cascades).

---

## 📞 Support

If you encounter issues during migration:
1. Check logs for specific error messages
2. Verify imports are correct
3. Ensure cleanup functions are defined
4. Test components individually before Trinity integration

**Your Trinity system will be bulletproof after this migration!** 🚀
