#!/usr/bin/env python3
"""Authoritative candidate check, runnable from ANY venv.

Reads candidate source on stdin, exits 0 if it compiles, 1 otherwise.
This is the process boundary that lets reactor's trainer consult a
validator living in a different virtualenv without importing it --
`REACTOR_GRPO_VERIFY_CMD` points here (or at a richer jarvis-side
validator that also runs the tests).
"""
import ast, sys
src = sys.stdin.read()
if not src.strip():
    sys.exit(1)
try:
    ast.parse(src)
except SyntaxError:
    sys.exit(1)
sys.exit(0)
