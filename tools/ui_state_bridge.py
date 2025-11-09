"""
Lightweight helper to push world-model snapshots and plan states to the UI demo.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


class UIStateBridge:
    """Best-effort HTTP publisher for robot_ui_demo panels."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 1.0) -> None:
        url = base_url or os.getenv("ROBOT_UI_URL") or "http://127.0.0.1:8000"
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self._world_failed = False
        self._plan_failed = False

    def post_world_model(self, snapshot: Dict[str, Any]) -> None:
        """Upload the latest world-model snapshot to the UI."""
        if not snapshot:
            return
        self._post(
            endpoint="/api/world_model/update",
            payload=snapshot,
            flag_attr="_world_failed",
        )

    def post_plan_state(
        self,
        *,
        root: Optional[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        current_index: int = -1,
        current_node: Optional[str] = None,
    ) -> None:
        """Upload the active behaviour tree and execution pointer."""
        payload = {
            "root": root,
            "steps": steps,
            "metadata": metadata or {},
            "current_index": current_index,
            "current_node": current_node,
        }
        self._post(
            endpoint="/api/plan/update",
            payload=payload,
            flag_attr="_plan_failed",
        )

    def _post(self, *, endpoint: str, payload: Dict[str, Any], flag_attr: str) -> None:
        url = f"{self.base_url}{endpoint}"
        try:
            requests.post(url, json=payload, timeout=self.timeout)
            setattr(self, flag_attr, False)
        except Exception as exc:  # noqa: BLE001
            # Only log the first failure to avoid spamming stdout.
            if not getattr(self, flag_attr):
                print(f"[UIStateBridge] POST {url} failed: {exc}")  # noqa: T201
            setattr(self, flag_attr, True)
