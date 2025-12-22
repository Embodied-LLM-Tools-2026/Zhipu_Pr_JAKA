"""
Lightweight JSONL trace logger for skill calls.

Each instance writes skill invocation records to a dedicated trace file,
one JSON object per line, and keeps a per-episode step counter.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional


class TraceLogger:
    """Append-only JSONL logger for execution traces."""

    def __init__(
        self,
        *,
        base_dir: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        self.base_dir = base_dir or os.getenv("TRACE_BASE_DIR", "runs")
        self.episode_id = episode_id or time.strftime("%Y%m%d_%H%M%S")
        self._lock = threading.Lock()
        self._step_counters: Dict[str, int] = {}
        self.file_path: Optional[str] = None
        self._enabled = True
        self._init_file()

    def _init_file(self) -> None:
        try:
            run_dir = os.path.join(self.base_dir, self.episode_id)
            os.makedirs(run_dir, exist_ok=True)
            self.file_path = os.path.join(run_dir, "trace.jsonl")
        except Exception:
            # Disable logging if filesystem is unavailable.
            self._enabled = False

    def _next_step(self, episode_id: str) -> int:
        current = self._step_counters.get(episode_id, 0) + 1
        self._step_counters[episode_id] = current
        return current

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Write a generic event to the trace."""
        if not self._enabled or self.file_path is None:
            return
        try:
            entry = {
                "type": event_type,
                "ts": time.time(),
                "payload": payload
            }
            with self._lock, open(self.file_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def log_skill_call(self, event: Dict[str, Any]) -> None:
        """Write a single skill call trace entry as JSONL."""
        if not self._enabled or self.file_path is None:
            return
        try:
            ep_id = str(event.get("episode_id") or self.episode_id)
            event.setdefault("episode_id", ep_id)
            event.setdefault("step_id", self._next_step(ep_id))
            with self._lock, open(self.file_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception:
            # Swallow any trace failures to avoid affecting main control flow.
            self._enabled = False
