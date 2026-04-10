#!/usr/bin/env python3
"""
FIPS Compliance Test Script

Verifies that the VisDrone pipeline makes zero network connections at runtime.
Works by monkey-patching socket.socket.connect to intercept any outbound call,
then exercising each pipeline component (ultralytics import, model load,
DeepSORT tracker init, and a dummy inference pass).

Usage:
    # Minimal (no model needed — tests imports and tracker only)
    python3 scripts/test_fips_compliance.py

    # Full (also loads a YOLO model and runs a dummy prediction)
    python3 scripts/test_fips_compliance.py --weights runs/detect/visdrone/weights/best.pt

Exit code 0 = all checks passed (FIPS-safe).
Exit code 1 = at least one check failed (network call detected).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import textwrap
from pathlib import Path

# ── Ensure YOLO_OFFLINE is set (mirrors what every real script does) ─────────
os.environ["YOLO_OFFLINE"] = "1"
os.environ["YOLO_SETTINGS_SYNC"] = "False"

# Add parent directory so visdrone_toolkit is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Network trap ─────────────────────────────────────────────────────────────
_original_connect = socket.socket.connect
_violations: list[tuple[str, tuple]] = []


def _trap_connect(self, address):
    """Record and block any outbound connection attempt."""
    _violations.append(("socket.connect", address))
    raise OSError(
        f"[FIPS-TEST] Blocked outbound connection to {address}"
    )


def _install_trap():
    _violations.clear()
    socket.socket.connect = _trap_connect  # type: ignore[assignment]


def _remove_trap():
    socket.socket.connect = _original_connect  # type: ignore[assignment]


# ── Test helpers ─────────────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []


def _run_check(name: str, fn):
    """Run *fn* with the network trap active and record pass/fail."""
    _install_trap()
    try:
        fn()
        passed = len(_violations) == 0
        detail = "no network calls" if passed else f"{len(_violations)} blocked call(s)"
        _results.append((name, passed, detail))
    except Exception as exc:
        # An exception that is NOT our trap means something else broke.
        if _violations:
            _results.append((name, False, f"blocked {len(_violations)} call(s); also raised {exc}"))
        else:
            _results.append((name, False, f"error: {exc}"))
    finally:
        _remove_trap()
        _violations.clear()


# ── Individual checks ────────────────────────────────────────────────────────

def check_ultralytics_import():
    """Importing ultralytics must not trigger any network I/O."""
    from ultralytics import YOLO  # noqa: F401


def check_model_load(weights_path: str):
    """Loading a YOLO model from local weights must not phone home."""
    from ultralytics import YOLO
    YOLO(weights_path)


def check_model_predict(weights_path: str):
    """Running a dummy prediction must not trigger any network call."""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(weights_path)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)


def check_tracker_init():
    """Initialising the DeepSORT tracker (MobileNet embedder) must be offline."""
    from visdrone_toolkit.tracker import DeepSORTTracker
    DeepSORTTracker(max_age=5, n_init=1)


def check_tracker_update():
    """Running a tracker update on a dummy frame must not use the network."""
    import numpy as np
    from visdrone_toolkit.tracker import DeepSORTTracker

    tracker = DeepSORTTracker(max_age=5, n_init=1)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    class_ids = np.array([0], dtype=np.int32)
    tracker.update(frame, boxes, scores, class_ids)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FIPS compliance test")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to YOLO weights (.pt). If omitted, model-load and predict checks are skipped.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FIPS Compliance Test — VisDrone Pipeline")
    print("=" * 60)
    print(f"  YOLO_OFFLINE = {os.environ.get('YOLO_OFFLINE', '<unset>')}")
    print()

    # Always run these
    _run_check("Import ultralytics", check_ultralytics_import)
    _run_check("DeepSORT tracker init", check_tracker_init)
    _run_check("DeepSORT tracker update", check_tracker_update)

    # Conditional on weights being available
    if args.weights:
        if not Path(args.weights).exists():
            print(f"  WARNING: weights file not found: {args.weights}")
            print("           Skipping model-load and predict checks.\n")
        else:
            _run_check("YOLO model load", lambda: check_model_load(args.weights))
            _run_check("YOLO model predict", lambda: check_model_predict(args.weights))
    else:
        print("  (--weights not provided; skipping model-load and predict checks)\n")

    # Report
    print("-" * 60)
    all_passed = True
    for name, passed, detail in _results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}]  {name:30s}  {detail}")
    print("-" * 60)

    if all_passed:
        print("\n  All checks passed — pipeline is FIPS-safe (no network calls).\n")
    else:
        print(textwrap.dedent("""
          FAILED — network calls were detected.
          Review the failing checks above and ensure YOLO_OFFLINE=1 is set
          before every ultralytics import.
        """))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
