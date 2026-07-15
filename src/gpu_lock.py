"""Serialize Intel iGPU / OpenVINO / Ultralytics inference across worker threads.

Running YOLO11 (OpenVINO) and YOLOPv2 (OpenVINO) on the same integrated GPU
from two daemon threads concurrently is a common cause of silent process
death after several seconds (driver TDR / OpenVINO native abort). One process-
wide lock keeps both models alive without changing their algorithms.
"""

from __future__ import annotations

import threading

INFERENCE_LOCK = threading.RLock()
