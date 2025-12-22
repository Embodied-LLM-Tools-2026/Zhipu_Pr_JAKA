"""
Function-call driven TaskProcessor.

This module introduces an alternative orchestration loop where an LLM
controls the robot purely via structured function calls. It runs
separately from the legacy behaviour tree pipeline so callers can select
either mode via configuration.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import base64
from zhipuai import ZhipuAI

try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    dashscope = None
    MultiModalConversation = None

from tools.vision.upload_image import upload_file_and_get_url

from tools.logging.task_logger import log_error, log_info, log_success, log_warning  # type: ignore
from tools.ui.ui_state_bridge import UIStateBridge  # type: ignore

from ...control.apis import RobotAPI
from ...control.world_model import WorldModel
from ...control.executor import SkillExecutor, SkillRuntime
from ...control.task_structures import ExecutionResult, FailureCode, ObservationPhase, PlanNode, ExecutionTurn, PlanContextEntry, InspectionPacket
from ...control.recovery_manager import RecoveryManager, RecoveryContext, RecoveryDecision
from ...agents.plan_runner import PlanRunner
from ...agents.task_plan.schema import validate_plan, PlanValidationError
from ..vlm_inspector import VLMInspector, InspectionReport

class FunctionCallToolset:
    """Wraps RobotAPI primitives as callable functions for the LLM."""

    def __init__(self, api: RobotAPI, processor: "FunctionCallTaskProcessor") -> None:
        self.api = api
        self.processor = processor

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches a tool call to the appropriate API method.
        Recovery is handled in FunctionCallTaskProcessor.invoke_skill.
        """
        if not hasattr(self.api, name):
            return {"ok": False, "error": f"unknown_function:{name}"}
        method = getattr(self.api, name)
        try:
            result = method(**args)
            if isinstance(result, ExecutionResult):
                return {
                    "ok": result.status == "success",
                    "status": result.status,
                    "reason": result.reason,
                    "evidence": result.evidence,
                }
            if isinstance(result, dict):
                return result
            return {"ok": True, "result": result}
        except Exception as exc:
            log_error(f"❌ Function {name} 调用失败: {exc}")
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    def specs(self) -> List[Dict[str, Any]]:
        specs = [
            {
                "name": "observe_scene",
                "description": "Capture a new observation of the target object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Object identifier or description."},
                        "phase": {
                            "type": "string",
                            "enum": ["search", "approach"],
                            "description": "Observation phase hint.",
                        },
                        "force_vlm": {"type": "boolean", "description": "Force a full VLM refresh."},
                        "query": {
                            "type": "string",
                            "description": "Optional natural-language request (e.g. describe scene relations).",
                        },
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "rotate_scan",
                "description": "Rotate the base in place to search the environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "angle_deg": {"type": "number", "description": "Rotation angle in degrees (default 30)."}
                    },
                },
            },
            {
                "name": "search_area",
                "description": "Perform repeated rotations/scans at the current spot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "turns": {"type": "integer", "description": "Number of rotations."},
                        "angle_deg": {"type": "number", "description": "Per-rotation angle in degrees."},
                    },
                },
            },
            {
                "name": "navigate_area",
                "description": "Navigate the base to a named area, marker, or explicit pose.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "area": {"type": "string"},
                        "marker": {"type": "string"},
                        "pose": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "theta": {"type": "number"},
                            },
                        },
                    },
                },
            },
            {
                "name": "get_object_state",
                "description": "Return the latest known state of an object from the world model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_id": {"type": "string", "description": "Object identifier (target name)."},
                    },
                    "required": ["object_id"],
                },
            },
            {
                "name": "navigate_to_pose",
                "description": "Command the base to an explicit pose (meters + radians).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "theta": {"type": "number", "description": "Rotation in radians."},
                    },
                    "required": ["x", "y", "theta"],
                },
            },
            {
                "name": "move_tcp",
                "description": "Perform a TCP motion (PTP via IK) using millimetre cartesian pose + axis-angle rotation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pose": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "[x_mm, y_mm, z_mm, rx, ry, rz]",
                        },
                        "speed": {"type": "number", "description": "Optional joint speed override."},
                        "acc": {"type": "number", "description": "Optional joint acceleration override."},
                    },
                    "required": ["pose"],
                },
            },
            {
                "name": "set_gripper",
                "description": "Update the gripper position/force.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position": {"type": "number", "description": "1-100 opening percentage."},
                        "force": {"type": "number", "description": "20-320 force value."},
                    },
                    "required": [],
                },
            },
            {
                "name": "execute_skill",
                "description": "Invoke a legacy skill from the SkillExecutor (e.g. open_gripper, finalize_target_pose).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill name."},
                        "args": {"type": "object", "description": "Arguments dict for the skill."},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "approach_far",
                "description": "Approach the target object when it is further than two metres away.",
                "parameters": {
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                },
            },
            {
                "name": "finalize_target_pose",
                "description": "Use depth localization + base alignment before grasping.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "predict_grasp_point",
                "description": "Run ZeroGrasp inference to obtain a grasp pose.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "execute_grasp",
                "description": "Execute the previously predicted grasp TCP pose.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "vla_grasp_finish",
                "description": "Short-horizon VLA tool to finish a grasp (1-3 seconds).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "description": "High-level intent for the VLA finisher."},
                        "image": {"type": "string", "description": "Optional image reference/URL."},
                        "state": {"type": "object", "description": "Optional robot state or context."},
                    },
                    "required": ["instruction"],
                },
            },
            {
                "name": "align_to_container",
                "description": "Align arm to container: Observe -> SAM Mask -> Center & Max Height -> Move to Center + MaxZ + 10cm. Orientation aligns with origin-to-center vector.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Container name (e.g. 'plate', 'box')."},
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "pick",
                "description": "High-level pick: Observe -> Predict -> Execute -> Close -> Lift. Assumes base is aligned.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target object name to pick."},
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "place",
                "description": "High-level place: Align to Container -> Lower 5cm -> Open -> Lift. Assumes holding object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Container name (e.g. 'plate', 'box')."},
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "open_gripper",
                "description": "Open the gripper via the legacy skill.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "close_gripper",
                "description": "Close the gripper via the legacy skill.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "handover_item",
                "description": "Perform a handover to a human once an item is grasped.",
                "parameters": {
                    "type": "object",
                    "properties": {"item": {"type": "string", "description": "Optional item name."}},
                },
            },
            {
                "name": "align_tcp_to_target",
                "description": "Move the robot arm (TCP) to a point 20cm back from the target object, aligned with the robot-object line.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target object name."},
                        "offset_mm": {"type": "number", "description": "Offset distance in mm (default 200)."},
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "vla_execute",
                "description": "Execute a VLA (Vision-Language-Action) skill with a natural language instruction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "description": "Instruction for the VLA model (e.g. 'pick up apple', 'open door')."},
                        "image": {"type": "string", "description": "Optional base64 image."},
                        "state": {"type": "object", "description": "Optional robot state."},
                    },
                    "required": ["instruction"],
                },
            },
            {
                "name": "pick_vla",
                "description": "Pick up an object using VLA for the final grasp (Align -> VLA Pick).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target object name."},
                        "instruction": {"type": "string", "description": "Optional VLA instruction."},
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "place_vla",
                "description": "Place an object into a container using VLA (Align -> VLA Place).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Container name."},
                        "instruction": {"type": "string", "description": "Optional VLA instruction."},
                    },
                    "required": ["target"],
                },
            },
        ]
        # Force disable execute_plan to ensure step-by-step VLM execution
        # if self.processor.enable_execute_plan_tool:
        #     specs.append(
        #         {
        #             "name": "execute_plan",
        #             "description": "Validate and execute a long-horizon plan JSON (macro-level fetch/place).",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "plan_json": {
        #                         "type": "string",
        #                         "description": "Plan JSON string matching the schema (goal, plan_id, subtasks).",
        #                     }
        #                 },
        #                 "required": ["plan_json"],
        #             },
        #         }
        #     )
        # {
        #     "name": "recover",
        #     "description": "Trigger the recovery skill (typically back off).",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "mode": {"type": "string"},
        #             "distance": {"type": "number"},
        #         },
        #     },
        # },
        return specs

    # ------------------------------------------------------------------
    def dispatch(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        handler = getattr(self, f"_fn_{name}", None)
        if not handler:
            return {"ok": False, "error": f"unknown_function:{name}"}
        try:
            return handler(**params)
        except Exception as exc:  # pragma: no cover - hardware path
            log_error(f"❌ Function {name} 调用失败: {exc}")
            return {"ok": False, "error": str(exc)}

    # -- individual function handlers ---------------------------------
    def _fn_observe_scene(
        self,
        target: str,
        phase: str = "search",
        force_vlm: bool = False,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        observation, payload = self.api.perception.observe(
            target,
            phase=ObservationPhase.SEARCH if phase != "approach" else ObservationPhase.APPROACH,
            force_vlm=force_vlm,
            analysis_request=query,
        )
        self.processor.update_observation(observation)
        return {
            "ok": True,
            "observation": self._serialize_observation(observation),
            "vlm_payload": payload,
        }

    def _fn_rotate_scan(self, angle_deg: Optional[float] = None) -> Dict[str, Any]:
        args = {"angle_deg": float(angle_deg)} if angle_deg is not None else {}
        return self._call_skill("rotate_scan", args)

    def _fn_search_area(self, turns: Optional[int] = None, angle_deg: Optional[float] = None) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        if turns is not None:
            args["turns"] = int(turns)
        if angle_deg is not None:
            args["angle_deg"] = float(angle_deg)
        return self._call_skill("search_area", args)

    def _fn_navigate_area(
        self,
        area: Optional[str] = None,
        marker: Optional[str] = None,
        pose: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        if area:
            args["area"] = area
        if marker:
            args["marker"] = marker
        if pose:
            args["pose"] = pose
        return self._call_skill("navigate_area", args)

    def _fn_get_object_state(self, object_id: str) -> Dict[str, Any]:
        state = self.api.perception.get_object_state(object_id)
        return {"ok": bool(state), "state": state}

    def _fn_navigate_to_pose(self, x: float, y: float, theta: float) -> Dict[str, Any]:
        obs_result = self._ensure_target_observation()
        success = self.api.navigation.goto_pose(x, y, theta)
        response = {
            "ok": success,
            "pose": {"x": x, "y": y, "theta": theta},
        }
        if obs_result and "observation" in obs_result:
            response["observation"] = obs_result["observation"]
        return response

    def _handle_recovery(
        self, result: ExecutionResult, runtime: SkillRuntime, node: PlanNode, inspection_report: InspectionReport = None
    ) -> ExecutionResult:
        """
        Centralized recovery handler.
        Uses VLMInspector to diagnose the failure, then RecoveryManager to suggest actions.
        """
        failure_code = result.failure_code or FailureCode.UNKNOWN
        log_warning(f"⚠️ [Recovery] Handling failure: {node.name} -> {failure_code.value}")

        report = inspection_report

        # If no report provided (e.g. hardware failure before inspection), run inspection now
        if not report:
            # 1. Capture Post-Execution Observation (The "Crime Scene" Photo)
            # We do this immediately so VLM sees the state right after failure.
            post_exec_obs = None
            try:
                # Force a quick observation (no VLM analysis yet, just capture)
                # We use the current target if available, or just a generic capture
                target = self.world.goal or "unknown"
                obs, _ = self.api.perception.observe(target, force_vlm=False)
                post_exec_obs = self._serialize_observation(obs)
                # Also update the main observation state since we just took a fresh one
                self.update_observation(obs, reset_extra=False)
            except Exception as e:
                log_warning(f"⚠️ [Recovery] Failed to capture post-execution observation: {e}")

            # 2. Build Inspection Packet
            packet = result.evidence.get("inspection_packet")
            if not packet:
                # Should have been created by executor, but fallback if missing
                packet = InspectionPacket(
                    episode_id=runtime.extra.get("episode_id"),
                    step_id=runtime.extra.get("step_id"),
                    skill_name=node.name,
                    skill_args=node.args,
                    exec_result={"status": result.status, "failure_code": failure_code.value, "reason": result.reason},
                    timestamp=time.time()
                )
            
            # Inject the fresh observation into the packet
            if post_exec_obs:
                packet.post_execution_observation = post_exec_obs

            # 3. Run VLM Inspection
            inspector = VLMInspector(api_key=self.api_key, model=self.model)
            report = inspector.inspect(packet)
            
            log_info(f"🕵️ [Inspector] Verdict: {report.verdict_hint}, Hazards: {len(report.hazards)}")
            if report.hazards:
                for h in report.hazards:
                    log_warning(f"  - {h.type}: {h.why}")

        # 4. Consult Recovery Manager
        ctx = RecoveryContext(
            skill_name=node.name,
            failure_code=failure_code,
            history=self.execution_history[-5:], # Last 5 steps
            world_state=self.world,
            inspection_report=report
        )
        
        suggestions = self.recovery_manager.suggest_recovery(ctx)
        
        # 5. Return result with suggestions (Do NOT execute recovery automatically)
        # The main LLM will decide what to do based on this evidence.
        if result.evidence is None:
            result.evidence = {}
            
        result.evidence["recovery_suggestions"] = [s.to_dict() for s in suggestions]
        # Ensure report is attached if it wasn't already
        if "inspection_report" not in result.evidence:
            result.evidence["inspection_report"] = report.to_dict()
        
        log_info(f"🚑 [Recovery] Generated {len(suggestions)} suggestions. Returning to LLM.")
        return result

    def _old_handle_recovery(self, result: ExecutionResult, runtime: SkillRuntime, node: PlanNode) -> ExecutionResult:
        """Invoke shared RecoveryManager for FC mode with budgets."""
        failure_code = result.failure_code or FailureCode.UNKNOWN
        budget = {
            "total": self._recovery_budget.get("total", 0),
            "per_code": self._recovery_budget.get("per_code", {}),
            "elapsed_s": time.time() - self._recovery_budget.get("start_time", time.time()),
        }
        history_tail = [{"node": node.name, "status": result.status, "failure_code": failure_code.value}]
        ctx = RecoveryContext(
            episode_id=self._current_plan_id,
            step_id=node.name,
            task_goal=self.world.goal,
            world_snapshot=None,
            history_tail=history_tail,
            budget=budget,
        )
        decision = self.recovery_manager.handle_failure(failure_code, ctx)

        # Update budget counters
        per_code = self._recovery_budget.setdefault("per_code", {})
        per_code[failure_code.value] = per_code.get(failure_code.value, 0) + 1
        self._recovery_budget["total"] = self._recovery_budget.get("total", 0) + 1

        if decision.kind == "EXECUTE_ACTIONS":
            recovery_success = True
            for idx, action in enumerate(decision.actions, start=1):
                rec_node = PlanNode(type="action", name=action["skill_name"], args=action.get("args", {}))
                rec_runtime = SkillRuntime(
                    navigator=self.navigator,
                    world_model=self.world,
                    observation=runtime.observation,
                    extra=dict(runtime.extra or {}),
                )
                rec_runtime.extra.update(
                    {
                        "recovery_level": decision.level,
                        "recovery_attempt_idx": idx,
                        "recovery_policy": decision.reason,
                        "recovery_triggered": True,
                    }
                )
                rec_result = self.executor.execute(rec_node, rec_runtime)
                self._record_timeline_event(
                    stage="recovery",
                    node=action["skill_name"],
                    status=rec_result.status,
                    detail=f"{decision.level}:{decision.reason}",
                    elapsed=rec_result.elapsed,
                )
                if rec_result.status != "success":
                    recovery_success = False
                    # annotate side effect
                    if result.evidence is None:
                        result.evidence = {}
                    result.evidence["recovery_side_effect_failure_code"] = rec_result.failure_code.value if rec_result.failure_code else None
                    break
            if recovery_success:
                # retry original
                retry_runtime = SkillRuntime(
                    navigator=self.navigator,
                    world_model=self.world,
                    observation=runtime.observation,
                    extra=dict(runtime.extra or {}),
                )
                retry_runtime.extra["recovery_level"] = decision.level
                retry_result = self.executor.execute(node, retry_runtime)
                if retry_result.evidence is None:
                    retry_result.evidence = {}
                retry_result.evidence["recovery_decision"] = decision.evidence
                return retry_result
            # recovery failed, return original failure annotated
            if result.evidence is None:
                result.evidence = {}
            result.evidence["recovery_decision"] = decision.evidence
            result.evidence["recovery_success"] = False
            return result

        if decision.kind == "ESCALATE_L3":
            if result.evidence is None:
                result.evidence = {}
            result.evidence["recovery_decision"] = decision.evidence
            result.reason = result.reason or "escalate_L3"
            return result

        # ABORT or unknown kind
        if result.evidence is None:
            result.evidence = {}
        result.evidence["recovery_decision"] = decision.evidence
        result.reason = result.reason or "recovery_abort"
        return result

    def _fn_move_tcp(
        self,
        pose: List[float],
        speed: Optional[float] = None,
        acc: Optional[float] = None,
    ) -> Dict[str, Any]:
        obs_result = self._ensure_target_observation(phase=ObservationPhase.APPROACH)
        result = self.api.manipulation.move_tcp(pose, speed=speed, acc=acc)
        if obs_result and "observation" in obs_result and isinstance(result, dict):
            result = dict(result)
            result["observation"] = obs_result["observation"]
        return result

    def _fn_set_gripper(
        self,
        position: Optional[float] = None,
        force: Optional[float] = None,
    ) -> Dict[str, Any]:
        obs_result = self._ensure_target_observation()
        result = self.api.manipulation.set_gripper(position=position, force=force)
        if obs_result and "observation" in obs_result and isinstance(result, dict):
            result = dict(result)
            result["observation"] = obs_result["observation"]
        return result

    def _fn_execute_skill(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._call_skill(name, args or {})

    def _fn_approach_far(self, target: Optional[str] = None) -> Dict[str, Any]:
        args = {"target": target} if target else {}
        return self._call_skill("approach_far", args)

    def _fn_finalize_target_pose(self) -> Dict[str, Any]:
        return self._call_skill("finalize_target_pose", {})

    def _fn_predict_grasp_point(self) -> Dict[str, Any]:
        return self._call_skill("predict_grasp_point", {})

    def _fn_execute_grasp(self) -> Dict[str, Any]:
        return self._call_skill("execute_grasp", {})

    def _fn_align_to_container(self, target: str) -> Dict[str, Any]:
        """
        Align arm to container:
        1. Observe target (VLM + SAM).
        2. Get surface points from observation (Robot Frame: X=Down, Y=Left, Z=Forward).
        3. Calculate center (y, z) and max height (min_x).
        4. Calculate orientation (yaw) from origin to center (in Y-Z plane).
        5. Move to (min_x - 100, center_y, center_z) with Rx=yaw.
        """
        import math
        import numpy as np

        # 1. Observe
        obs_result = self._ensure_target_observation(target=target)
        if not obs_result:
             return {"ok": False, "status": "failed", "reason": f"Failed to observe target: {target}"}
        
        # 2. Get Surface Points
        latest_obs = self.processor._latest_observation
        if not latest_obs:
             return {"ok": False, "status": "failed", "reason": f"No observation found for target: {target}"}
        
        depth_info = self.api.manipulation._executor.localize_observation(latest_obs)
        if not depth_info:
             return {"ok": False, "status": "failed", "reason": "Failed to localize target depth."}
             
        surface_points_cam = depth_info.get("surface_points")
        if surface_points_cam is None or len(surface_points_cam) == 0:
             center_cam = depth_info.get("obj_center_3d")
             if not center_cam:
                 return {"ok": False, "status": "failed", "reason": "No 3D points or center found."}
             surface_points_cam = [center_cam]
             
        # Transform to Robot Frame (X=Down, Y=Left, Z=Forward)
        surface_points_robot = []
        for pt in surface_points_cam:
            pt_mm = np.array(pt, dtype=float)
            pt_robot = self.api.manipulation._executor.transform_camera_to_robot(pt_mm)
            surface_points_robot.append(pt_robot)
            
        surface_points_robot = np.array(surface_points_robot) # shape (N, 3)
        
        # 3. Calculate Center and Max Height
        # Robot Frame: X is Vertical Down. Y is Horizontal Left. Z is Horizontal Forward.
        # "Highest" point means minimum X (closest to base/ceiling).
        min_vals = np.min(surface_points_robot, axis=0)
        max_vals = np.max(surface_points_robot, axis=0)
        
        # Center in Horizontal Plane (Y-Z)
        center_y = (min_vals[1] + max_vals[1]) / 2.0
        center_z = (min_vals[2] + max_vals[2]) / 2.0
        
        # Max Height (Top of container) = Min X
        top_x = min_vals[0]
        
        # 4. Calculate Orientation (Yaw)
        # Direction from origin to center in Y-Z plane
        # yaw = atan2(y, z) -> Angle from Z axis towards Y axis
        yaw = math.atan2(center_y, center_z)
        
        # 5. Move to Target
        # Target X = top_x - 100mm (10cm above)
        target_x = top_x - 100.0
        
        # Get current pose to preserve Ry, Rz (assuming they control "Down" pointing)
        try:
            curr_pos, curr_euler = self.api.manipulation._executor.get_current_tool_pose()
            # curr_euler is [rx, ry, rz]
            target_rx = yaw
            target_ry = curr_euler[1]
            target_rz = curr_euler[2]
        except Exception:
            # Fallback if get_pose fails: assume standard down pointing?
            # But we don't know what "standard" is for this user.
            # Let's default to 0 for others if we fail, but logging error is better.
            log_warning("⚠️ [AlignContainer] Failed to get current pose, using 0 for Ry/Rz.")
            target_rx = yaw
            target_ry = 0.0
            target_rz = 0.0

        pose = [target_x, center_y, center_z, target_rx, target_ry, target_rz]
        
        log_info(f"🎯 [AlignContainer] Target: {target}, CenterYZ: ({center_y:.1f}, {center_z:.1f}), TopX: {top_x:.1f}, Yaw: {math.degrees(yaw):.1f}deg")
        
        result = self.api.manipulation.move_tcp(pose, speed=30.0)
        
        if result.get("status") == "success":
             return {"ok": True, "status": "success", "reason": "Aligned to container.", "pose": pose}
        else:
             return result

    def _fn_pick(self, target: str) -> Dict[str, Any]:
        """
        High-level pick function: Observe -> Predict -> Execute -> Close -> Lift.
        Assumes base is already aligned.
        """
        # 1. Observe
        obs_result = self._ensure_target_observation(target=target)
        if not obs_result:
             return {"ok": False, "status": "failed", "reason": f"Failed to observe target: {target}"}
        
        # 2. Predict Grasp Point
        pred_result = self._call_skill("predict_grasp_point", {})
        if not pred_result.get("ok"):
            return pred_result
            
        # 3. Execute Grasp (Move to grasp pose)
        exec_result = self._call_skill("execute_grasp", {})
        if not exec_result.get("ok"):
            return exec_result
            
        # 4. Close Gripper
        close_result = self._call_skill("close_gripper", {})
        if not close_result.get("ok"):
            return close_result
            
        # 5. Lift (Shift TCP up by 10cm)
        try:
            # We need current pose to calculate shift
            # get_current_tool_pose returns (pos_m, euler_rad) as numpy arrays
            pos, euler = self.api.manipulation._executor.get_current_tool_pose()
            current_pose = list(pos) + list(euler)
            
            lift_result = self.api.manipulation.shift_tcp(
                base_pose=current_pose,
                delta_xyz=[-0.1, 0.0, 0.0], # Up 10cm (X is down, so -X is up)
                speed=30.0,
                acc=50.0
            )
            if lift_result.get("status") != "success":
                 return {"ok": False, "status": "failed", "reason": f"Lift failed: {lift_result.get('reason')}"}
        except Exception as e:
             return {"ok": False, "status": "failed", "reason": f"Lift exception: {e}"}

        return {"ok": True, "status": "success", "reason": "Pick sequence completed successfully."}

    def _fn_place(self, target: str) -> Dict[str, Any]:
        """
        High-level place function:
        1. Align to container (Move to 10cm above).
        2. Lower to 5cm above container top.
        3. Open gripper.
        4. Lift back to 10cm above.
        """
        # 1. Align to Container (This handles observation, calculation, and move to +10cm)
        align_res = self._fn_align_to_container(target)
        if not align_res.get("ok"):
            return align_res
            
        # align_res should contain the pose we moved to
        # pose = [x, y, z, rx, ry, rz]
        # We moved to top_x - 100.0
        
        # 2. Lower to 5cm above (Move down by 5cm)
        # Current X is (top_x - 100). We want (top_x - 50).
        # So we need to increase X by 50mm (since X is down).
        
        # Use shift_tcp for relative move
        # delta_xyz = [50.0, 0.0, 0.0] (mm) -> But shift_tcp usually takes meters or mm?
        # Let's check shift_tcp signature in apis.py / executor.py
        # primitive_shift_tcp takes delta_xyz in METERS usually?
        # Let's check executor.py primitive_shift_tcp
        
        # Executor primitive_shift_tcp:
        # target_pos = current_pos + np.array(delta_xyz)
        # current_pos is in METERS (get_current_tool_pose returns meters).
        # So delta_xyz should be in METERS.
        
        # We want to move down 5cm = 0.05m.
        # X is down. So +0.05m in X.
        
        res = self.api.manipulation.shift_tcp(
            base_pose=None, # Will fetch current
            delta_xyz=[0.05, 0.0, 0.0], # Down 5cm
            speed=20.0
        )
        if res.get("status") != "success":
             return {"ok": False, "status": "failed", "reason": f"Lower failed: {res.get('reason')}"}
             
        # 3. Open Gripper
        res = self.api.manipulation.set_gripper(position=100)
        if res.get("status") != "success":
             return {"ok": False, "status": "failed", "reason": f"Open gripper failed: {res.get('reason')}"}
             
        # 4. Lift (Retreat)
        # Move up by 10cm = -0.1m in X.
        res = self.api.manipulation.shift_tcp(
            base_pose=None,
            delta_xyz=[-0.1, 0.0, 0.0], # Up 10cm
            speed=50.0
        )
        
        return {"ok": True, "status": "success", "reason": "Place sequence completed."}

    def _fn_vla_execute(
        self,
        instruction: str,
        image: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args: Dict[str, Any] = {"instruction": instruction}
        if image:
            args["image"] = image
        if state:
            args["state"] = state
        return self._call_skill("vla_execute", args)

    def _fn_pick_vla(self, target: str, instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Pick up an object using VLA for the final grasp.
        1. Align TCP to target (Approach).
        2. Execute VLA grasp.
        """
        # 1. Align TCP
        align_res = self._fn_align_tcp_to_target(target)
        if not align_res.get("ok"):
            return align_res
            
        # 2. VLA Execute
        instr = instruction or f"pick up {target}"
        vla_res = self._fn_vla_execute(instruction=instr)
        
        if vla_res.get("ok"):
             return {"ok": True, "status": "success", "reason": "Pick VLA sequence completed."}
        else:
             return vla_res

    def _fn_place_vla(self, target: str, instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an object into a container using VLA.
        1. Align to container (Approach).
        2. Execute VLA place.
        """
        # 1. Align to Container
        align_res = self._fn_align_to_container(target)
        if not align_res.get("ok"):
            return align_res
            
        # 2. VLA Execute
        instr = instruction or f"place object into {target}"
        vla_res = self._fn_vla_execute(instruction=instr)
        
        if vla_res.get("ok"):
             return {"ok": True, "status": "success", "reason": "Place VLA sequence completed."}
        else:
             return vla_res

    def _fn_open_gripper(self) -> Dict[str, Any]:
        return self._call_skill("open_gripper", {})

    def _fn_close_gripper(self) -> Dict[str, Any]:
        return self._call_skill("close_gripper", {})

    def _fn_handover_item(self, item: Optional[str] = None) -> Dict[str, Any]:
        args = {"item": item} if item else {}
        return self._call_skill("handover_item", args)

    def _fn_align_tcp_to_target(self, target: str, offset_mm: float = 200.0) -> Dict[str, Any]:
        """
        Align TCP to target:
        1. Observe & Localize target.
        2. Calculate center (x, y, z) in Robot Frame (X=Down, Y=Left, Z=Forward).
        3. Calculate yaw = atan2(y, z).
        4. Move to position retracted by offset_mm along the radial line from base to object.
           Target Y = y - offset * sin(yaw)
           Target Z = z - offset * cos(yaw)
           Target X = x (Keep object height)
        5. Orientation: Rx=pi (Down), Ry=0, Rz=yaw (Aligned with radius).
        """
        import math
        import numpy as np

        # 1. Observe
        obs_result = self._ensure_target_observation(target=target)
        if not obs_result:
             return {"ok": False, "status": "failed", "reason": f"Failed to observe target: {target}"}
        
        # 2. Localize
        latest_obs = self.processor._latest_observation
        if not latest_obs:
             return {"ok": False, "status": "failed", "reason": f"No observation found for target: {target}"}
        
        depth_info = self.api.manipulation._executor.localize_observation(latest_obs)
        if not depth_info:
             return {"ok": False, "status": "failed", "reason": "Failed to localize target depth."}
             
        center_cam = depth_info.get("obj_center_3d")
        if not center_cam:
             return {"ok": False, "status": "failed", "reason": "No 3D center found."}
             
        # Transform to Robot Frame (X=Down, Y=Left, Z=Forward)
        center_robot = self.api.manipulation._executor.transform_camera_to_robot(np.array(center_cam, dtype=float))
        x, y, z = center_robot # mm
        
        # 3. Calculate Orientation (Yaw)
        # Yaw in Y-Z plane (Horizontal)
        yaw = math.atan2(y, z)
        
        # 4. Calculate Target Position (Retracted)
        # Direction vector is (sin(yaw), cos(yaw))
        # We want to move BACK from the object towards the origin
        # Pos = Center - Offset * Direction
        
        target_y = y - offset_mm * math.sin(yaw)
        target_z = z - offset_mm * math.cos(yaw)
        target_x = x # Keep same height as object center
        
        # 5. Orientation
        # Downward pointing, aligned with radius
        target_rx = math.pi
        target_ry = 0.0
        target_rz = yaw
        
        pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
        
        log_info(f"🎯 [AlignTCP] Target: {target}, Center: ({x:.1f}, {y:.1f}, {z:.1f}), Yaw: {math.degrees(yaw):.1f}deg, Offset: {offset_mm}mm")
        
        result = self.api.manipulation.move_tcp(pose, speed=30.0)
        
        if result.get("status") == "success":
             return {"ok": True, "status": "success", "reason": "Aligned TCP to target.", "pose": pose}
        else:
             return result

    def _fn_recover(self, mode: Optional[str] = None, distance: Optional[float] = None) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        if mode:
            args["mode"] = mode
        if distance is not None:
            args["distance"] = float(distance)
        return self._call_skill("recover", args)

    def _fn_execute_plan(self, plan_json: Any) -> Dict[str, Any]:
        """Validate + execute a plan via the lightweight PlanRunner (gated by flag)."""
        if not self.processor.enable_execute_plan_tool:
            return {"ok": False, "error": "execute_plan_disabled"}
        return self.processor.execute_plan(plan_json)

    @staticmethod
    def _serialize_observation(observation: Any) -> Dict[str, Any]:
        if observation is None:
            return {}
        return {
            "found": bool(getattr(observation, "found", False)),
            "confidence": getattr(observation, "confidence", None),
            "bbox": getattr(observation, "bbox", None),
            "range_estimate": getattr(observation, "range_estimate", None),
            "camera_center": getattr(observation, "camera_center", None),
            "robot_center": getattr(observation, "robot_center", None),
            "world_center": getattr(observation, "world_center", None),
            "source": getattr(observation, "source", None),
            "analysis": getattr(observation, "analysis", None),
        }

    def _call_skill(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Always refresh observation before executing a skill to keep state current.
        # Note: Some skills (like pick) handle observation internally, but for generic calls we ensure it.
        # If the skill provides a specific target in args, we should observe that.
        target = args.get("target")
        obs_result = self._ensure_target_observation(target=target)
        result = self.processor.invoke_skill(name, args)
        response = {
            "ok": result.status == "success",
            "status": result.status,
            "reason": result.reason,
            "evidence": result.evidence,
        }
        if obs_result and "observation" in obs_result:
            response["observation"] = obs_result["observation"]
        return response

    def _ensure_target_observation(
        self, target: Optional[str] = None, phase: ObservationPhase = ObservationPhase.SEARCH
    ) -> Optional[Dict[str, Any]]:
        """Refresh VLM observation for the specified target (or current goal), if available."""
        target_to_observe = target or self.processor.world.goal
        if not target_to_observe:
            return None
            
        # Optimization: If observation is fresh and valid for THIS target, reuse it
        # We need to check if the latest observation was actually for this target
        latest_obs = self.processor._latest_observation
        is_same_target = latest_obs and getattr(latest_obs, "target_id", "") == target_to_observe
        
        if not self.processor._observation_stale and is_same_target and latest_obs:
            # Check if observation actually found the target (if not, we might want to retry)
            if latest_obs.found:
                return {
                    "observation": self._serialize_observation(latest_obs),
                    "vlm_payload": None # Payload not cached, but usually not needed for skills
                }

        try:
            observation, payload = self.api.perception.observe(
                target_to_observe, phase=phase, force_vlm=False, max_steps=1
            )
            # Ensure target_id is set on observation so we can verify it later
            if not hasattr(observation, "target_id"):
                setattr(observation, "target_id", target_to_observe)
                
            self.processor.update_observation(observation, reset_extra=False)
            return {
                "observation": self._serialize_observation(observation),
                "vlm_payload": payload,
            }
        except Exception as exc:
            log_warning(f"[FunctionCall] 自动观测目标 {target_to_observe} 失败: {exc}")
            return None


class FunctionCallTaskProcessor:
    """
    Alternative orchestration pipeline driven by LLM function calls.

    The processor exposes RobotAPI primitives to the LLM. The agent must
    call these functions (observe/navigate/manipulate) to achieve the
    requested goal. Final responses are delivered once the LLM returns a
    normal assistant message.
    """

    def __init__(self, navigator=None) -> None:
        self.navigator = navigator
        self.world = WorldModel()
        self.observer = VLMObserver()
        self.executor = SkillExecutor(navigator=navigator)
        # Planner/Reflection are still constructed to keep RobotAPI happy,
        # but are unused in this chain.
        self.planner = BehaviorPlanner()
        self.reflection = ReflectionAdvisor(
            llm_api_key=getattr(self.planner, "llm_api_key", None),
            llm_model=getattr(self.planner, "llm_model", "deepseek-chat"),
        )
        self.api = RobotAPI.build(
            navigator=self.navigator,
            observer=self.observer,
            executor=self.executor,
            planner=self.planner,
            world=self.world,
            reflection=self.reflection,
        )
        self.tools = FunctionCallToolset(self.api, self)
        self.recovery_manager = RecoveryManager()
        self.enable_execute_plan_tool = os.getenv("ENABLE_EXECUTE_PLAN_TOOL", "false").lower() not in {"0", "false", "no"}
        self.plan_runner: Optional[PlanRunner] = None
        if self.enable_execute_plan_tool:
            self.plan_runner = PlanRunner(
                self.executor,
                self.recovery_manager,
                self.world,
                max_tool_calls=int(os.getenv("PLAN_MAX_TOOL_CALLS", "50")),
                max_time_s=float(os.getenv("PLAN_MAX_TIME_S", "300")),
            )
        self.max_rounds = int(os.getenv("FUNCTION_CALL_MAX_ROUNDS", "16"))
        self.temperature = float(os.getenv("FUNCTION_CALL_TEMPERATURE", "0.1"))
        self._finalize_calls = 0
        self._recovery_budget = {
            "total": 0,
            "per_code": {},
            "start_time": time.time(),
        }
        
        # 模型提供商选择: "zhipu" (GLM-4.5v) 或 "dashscope" (Qwen3-VL)
        self.provider = os.getenv("FUNCTION_CALL_PROVIDER", "zhipu").lower()
        
        if self.provider == "dashscope":
            # Qwen3-VL via DashScope
            self.model = os.getenv("FUNCTION_CALL_MODEL", "qwen-vl-max")
            self.api_key = (
                os.getenv("FUNCTION_CALL_API_KEY")
                or os.getenv("DASHSCOPE_API_KEY")
                or os.getenv("Zhipu_real_demo_API_KEY")  # 复用 VLM 的 key
            )
            if not self.api_key:
                raise RuntimeError("FunctionCallTaskProcessor (dashscope) 需要配置 DASHSCOPE_API_KEY")
            if not DASHSCOPE_AVAILABLE:
                raise RuntimeError("dashscope 库未安装，请运行: pip install dashscope")
            dashscope.api_key = self.api_key
            self.client = None  # dashscope 不需要 client 对象
            log_info(f"[FunctionCall] 使用 DashScope 模型: {self.model}")
        else:
            # GLM-4.5v via ZhipuAI (default)
            self.provider = "zhipu"
            self.model = os.getenv("FUNCTION_CALL_MODEL", "glm-4.5v")
            self.api_key = (
                os.getenv("FUNCTION_CALL_API_KEY")
                or os.getenv("ZHIPUAI_API_KEY")
                or getattr(self.planner, "llm_api_key", None)
            )
            if not self.api_key:
                raise RuntimeError("FunctionCallTaskProcessor (zhipu) 需要配置 ZHIPUAI_API_KEY")
            self.client = ZhipuAI(api_key=self.api_key)
            log_info(f"[FunctionCall] 使用 ZhipuAI 模型: {self.model}")
        
        self.system_prompt = os.getenv(
            "FUNCTION_CALL_SYSTEM_PROMPT",
            (
                "You are a robotics planner controlling a real robot. "
                "CRITICAL RULES:\n"
                "1. You MUST use tool_calls (function calls) to execute any action. NEVER describe actions in text.\n"
                "2. Do NOT say 'I will call predict_grasp_point' - instead, actually call the function.\n"
                "3. Do NOT include function names in your text response. Use the tools directly.\n"
                "4. Only respond with a final JSON summary AFTER the grasp is complete or truly impossible.\n"
                "5. Never fabricate execution results - always call the actual function and wait for results.\n"
                "If you need to predict a grasp point, call predict_grasp_point(). If you need to execute grasp, call execute_grasp()."
                + ("\nIf the user requests long-horizon tasks or sets output_plan=true, produce a Plan JSON and call execute_plan(plan_json) instead of low-level skill calls." if self.enable_execute_plan_tool else "")
            ),
        )
        self._latest_observation: Optional[Any] = None
        self._shared_runtime_extra: Dict[str, Any] = {}
        disable_bridge = os.getenv("DISABLE_ROBOT_UI_BRIDGE", "").lower() in {"1", "true", "yes"}
        self.ui_bridge = None if disable_bridge else UIStateBridge(os.getenv("ROBOT_UI_URL"))
        self._timeline: List[Dict[str, Any]] = []
        self._timeline_limit = int(os.getenv("EXEC_TIMELINE_LIMIT", "30"))
        self.execution_history: List[ExecutionTurn] = []
        self._execution_history_limit = int(os.getenv("EXEC_HISTORY_LIMIT", "60"))
        self._current_plan_entry: Optional[PlanContextEntry] = None
        self._current_plan_id: Optional[str] = None
        self._last_status_message: Optional[str] = None
        self._mission_text = self._build_mission_instruction()
        self._tool_instruction = self._build_tool_instruction()
        self._failure_threshold = int(os.getenv("FUNCTION_CALL_FAILURE_THRESHOLD", "3"))
        self._failure_count = 0
        self._observation_stale = True  # Initially stale until first observation

    def _get_current_image_base64(self, cam_name: str = "front") -> Optional[str]:
        """获取当前相机图像，根据 provider 返回不同格式"""
        try:
            # Capture image from local API
            resp = requests.get(
                f"http://127.0.0.1:8000/api/capture?cam={cam_name}",
                timeout=3,
            )
            if resp.status_code != 200:
                log_warning(f"Failed to capture image: {resp.status_code}")
                return None
            
            data = resp.json()
            image_path = data.get("url")
            if not image_path or not os.path.exists(image_path):
                log_warning(f"Image path invalid: {image_path}")
                return None
            
            # 根据 provider 返回不同格式
            if self.provider == "dashscope":
                # DashScope 需要上传图像到 OSS 获取 oss:// URL
                try:
                    oss_url = upload_file_and_get_url(
                        api_key=self.api_key,
                        model_name=self.model,
                        file_path=image_path,
                    )
                    log_info(f"[FunctionCall] 图像已上传到 OSS: {oss_url[:50]}...")
                    return oss_url
                except Exception as e:
                    log_error(f"上传图像到 OSS 失败: {e}")
                    return None
            else:
                # ZhipuAI 使用 base64 格式
                with open(image_path, "rb") as img_file:
                    img_base = base64.b64encode(img_file.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{img_base}"
        except Exception as e:
            log_error(f"Error getting camera image: {e}")
            return None

    # ------------------------------------------------------------------
    def set_navigator(self, navigator) -> None:
        self.navigator = navigator
        self.executor.set_navigator(navigator)
        self.api.update_navigator(navigator)

    def update_observation(self, observation: Any, reset_extra: bool = True) -> None:
        self._latest_observation = observation
        self._observation_stale = False
        if reset_extra:
            self._shared_runtime_extra = {}

    def invoke_skill(self, name: str, args: Dict[str, Any]) -> ExecutionResult:
        runtime_extra = dict(self._shared_runtime_extra or {})
        if self._current_plan_id and "episode_id" not in runtime_extra:
            runtime_extra["episode_id"] = self._current_plan_id
        runtime = SkillRuntime(
            navigator=self.navigator,
            world_model=self.world,
            observation=self._latest_observation,
            extra=runtime_extra,
        )
        node = PlanNode(type="action", name=name, args=args or {})
        
        # 1. Execute Skill (includes internal hardware verifiers)
        result = self.executor.execute(node, runtime)
        
        # 2. Invalidate observation if skill is mutating (moves robot or arm)
        if result.status == "success" and self._is_mutating_skill(name):
            self._observation_stale = True
            
        # 3. Post-Execution Inspection (Visual Verifier)
        # Always run VLM Inspector for every skill to verify outcome visually
        
        # 3.1 Capture "Crime Scene" / "Success Scene" photo
        post_exec_obs = None
        try:
            target = self.world.goal or "unknown"
            # force_vlm=False -> RGB only, fast
            obs, _ = self.api.perception.observe(target, force_vlm=False)
            post_exec_obs = self._serialize_observation(obs)
            # Also update the main observation state since we just took a fresh one
            self.update_observation(obs, reset_extra=False)
        except Exception as e:
            log_warning(f"⚠️ [invoke_skill] Failed to capture post-execution observation: {e}")

        # 3.2 Prepare Packet
        packet = result.evidence.get("inspection_packet")
        if not packet:
            # Should have been created by executor, but fallback if missing
            packet = InspectionPacket(
                episode_id=runtime.extra.get("episode_id"),
                step_id=runtime.extra.get("step_id"),
                skill_name=node.name,
                skill_args=node.args,
                exec_result={"status": result.status, "failure_code": result.failure_code.value if result.failure_code else None, "reason": result.reason},
                timestamp=time.time()
            )
        
        if post_exec_obs:
            packet.post_execution_observation = post_exec_obs

        # 3.3 Run VLM Inspector
        inspector = VLMInspector(api_key=self.api_key, model=self.model)
        report = inspector.inspect(packet)
        
        # Log inspection result
        log_info(f"🕵️ [Inspector] Verdict: {report.verdict_hint}, Hazards: {len(report.hazards)}")
        if report.hazards:
            for h in report.hazards:
                log_warning(f"  - {h.type}: {h.why}")
        
        # Attach report to result
        if result.evidence is None: result.evidence = {}
        result.evidence["inspection_report"] = report.to_dict()

        # 3.4 Override Status if VLM detects failure
        if result.status == "success" and report.verdict_hint == "FAIL":
            log_warning(f"⚠️ [invoke_skill] Skill succeeded but VLM Inspector reported FAIL. Overriding status.")
            result.status = "failure"
            result.failure_code = FailureCode.VERIFICATION_FAILED
            result.reason = "vlm_verification_failed"
            # Add hazards to reason for clarity
            if report.hazards:
                result.reason += f": {report.hazards[0].why}"

        # 4. Handle Recovery (Generate Suggestions) if failed
        if result.status == "failure":
            # We pass the report we just generated to _handle_recovery
            # so it doesn't run inspection again.
            result = self._handle_recovery(result, runtime, node, inspection_report=report)

        # Runtime.extra may be mutated by skills (e.g. predict_grasp_point), reuse it for future calls.
        self._shared_runtime_extra = runtime.extra or self._shared_runtime_extra
        return result

    def _is_mutating_skill(self, name: str) -> bool:
        """Check if skill changes robot state, requiring new observation."""
        # Navigation, Manipulation, and Scanning invalidate the view
        mutating_prefixes = [
            "navigate", "move", "rotate", "turn", 
            "pick", "place", "grasp", "release", "open", "close",
            "home", "approach", "align"
        ]
        return any(name.startswith(p) for p in mutating_prefixes)

    def execute_plan(self, plan_json: Any) -> Dict[str, Any]:
        """
        Lightweight plan execution entry used by the execute_plan tool.
        Returns a structured payload suitable for LLM consumption.
        """
        if not self.enable_execute_plan_tool or not self.plan_runner:
            return {"ok": False, "error": "execute_plan_disabled"}

        # Accept raw dict or JSON string
        raw_plan = plan_json
        if isinstance(plan_json, str):
            try:
                raw_plan = json.loads(plan_json)
            except Exception as exc:
                return {
                    "ok": False,
                    "type": "VALIDATION_ERROR",
                    "errors": [f"invalid_json:{exc}"],
                    "plan_id": None,
                }

        if not isinstance(raw_plan, dict):
            return {
                "ok": False,
                "type": "VALIDATION_ERROR",
                "errors": ["plan_json must be an object or JSON string"],
                "plan_id": None,
            }

        try:
            plan = validate_plan(raw_plan)
        except PlanValidationError as exc:
            return {
                "ok": False,
                "type": "VALIDATION_ERROR",
                "errors": [str(exc)],
                "plan_id": raw_plan.get("plan_id"),
            }

        # Prepare runtime context
        prev_plan_id = self._current_plan_id
        self._current_plan_id = plan.plan_id
        self.world.set_goal(plan.goal)
        runtime = SkillRuntime(
            navigator=self.navigator,
            world_model=self.world,
            observation=self._latest_observation,
            extra={"episode_id": plan.plan_id, "plan_id": plan.plan_id},
        )

        runner_result = self.plan_runner.run(plan, runtime)
        self._current_plan_id = prev_plan_id

        return {
            "ok": runner_result.ok,
            "plan_id": runner_result.plan_id,
            "subtasks_status": runner_result.subtasks_status,
            "final_outcome": runner_result.final_outcome,
            "last_failure": runner_result.last_failure,
            "trace_pointer": runner_result.trace_pointer,
            "reason": runner_result.reason,
        }

    def _publish_world_snapshot(self) -> None:
        if not self.ui_bridge:
            return
        try:
            snapshot = self.world.snapshot()
        except Exception:
            return
        self.ui_bridge.post_world_model(snapshot)

    def _emit_ui_log(self, message: str, level: str = "info") -> None:
        if not self.ui_bridge:
            return
        try:
            self.ui_bridge.post_task_log(message, level=level)
        except Exception:
            pass

    def _record_timeline_event(
        self,
        *,
        stage: str,
        node: str,
        status: str,
        detail: Optional[str] = None,
        elapsed: Optional[float] = None,
    ) -> None:
        entry = {
            "ts": int(time.time() * 1000),
            "time": time.strftime("%H:%M:%S"),
            "stage": stage,
            "node": node,
            "status": status,
            "detail": detail or "",
        }
        if elapsed is not None:
            entry["elapsed"] = float(elapsed)
        self._timeline.append(entry)
        if len(self._timeline) > self._timeline_limit:
            self._timeline = self._timeline[-self._timeline_limit :]

    def _timeline_payload(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._timeline]

    def _append_execution_turn(
        self,
        *,
        stage: str,
        node: str,
        status: str,
        detail: Optional[str] = None,
    ) -> None:
        if not self._current_plan_id:
            return
        turn = ExecutionTurn(
            plan_id=self._current_plan_id,
            stage=stage,
            node=node,
            status=status,
            detail=detail,
        )
        self.execution_history.append(turn)
        if len(self.execution_history) > self._execution_history_limit:
            self.execution_history = self.execution_history[-self._execution_history_limit :]

    # ------------------------------------------------------------------
    def _call_llm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.provider == "dashscope":
            return self._call_llm_dashscope(messages)
        else:
            return self._call_llm_zhipu(messages)
    
    def _call_llm_zhipu(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用 ZhipuAI GLM-4.5v"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": spec} for spec in self.tools.specs()],
                tool_choice="auto",
                temperature=self.temperature,
            )
            return response.choices[0].message.model_dump()
        except Exception as e:
            log_error(f"ZhipuAI LLM call failed: {e}")
            raise RuntimeError(f"function_call_llm_failed: {e}")
    
    def _call_llm_dashscope(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用 DashScope Qwen3-VL (支持 function call)"""
        try:
            if not MultiModalConversation:
                raise RuntimeError("dashscope.MultiModalConversation 未安装")
            # 转换消息格式为 DashScope 格式
            ds_messages = self._convert_messages_for_dashscope(messages)
            
            # 构建 tools 参数
            tools = [{"type": "function", "function": spec} for spec in self.tools.specs()]
            
            response = MultiModalConversation.call(
                model=self.model,
                messages=ds_messages,
                tools=tools,
                result_format="message",
                temperature=self.temperature,
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"DashScope API error: {response.code} - {response.message}")
            
            # 解析响应
            output = response.output
            choice = output.choices[0] if output.choices else None
            if not choice:
                raise RuntimeError("DashScope 返回空响应")
            
            message = choice.message
            result = {
                "role": message.role,
                "content": message.content or "",
            }
            
            # 处理 tool_calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                result["tool_calls"] = []
                for tc in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.get("id") or tc.get("function", {}).get("name", "call_0"),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        }
                    })
            
            return result
        except Exception as e:
            log_error(f"DashScope LLM call failed: {e}")
            raise RuntimeError(f"function_call_llm_failed: {e}")
    
    def _convert_messages_for_dashscope(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将消息格式转换为 DashScope 兼容格式"""
        ds_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            
            # 处理 tool 消息
            if role == "tool":
                ds_messages.append({
                    "role": "tool",
                    "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                    "tool_call_id": msg.get("tool_call_id", ""),
                })
                continue
            
            # 处理 assistant 消息 (可能包含 tool_calls)
            if role == "assistant":
                ds_msg = {"role": "assistant", "content": msg.get("content") or ""}
                if "tool_calls" in msg and msg["tool_calls"]:
                    ds_msg["tool_calls"] = msg["tool_calls"]
                ds_messages.append(ds_msg)
                continue
            
            # 处理 user/system 消息 (可能包含图像)
            if isinstance(content, list):
                # 多模态内容: DashScope 需要 {"image": "..."} / {"text": "..."} 形式
                ds_content = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type == "text" or ("text" in part and part_type is None):
                        ds_content.append({"text": part.get("text", "")})
                        continue
                    # 兼容 OpenAI 风格的 image_url 结构
                    if part_type == "image_url" or "image_url" in part or "image" in part:
                        image_url = ""
                        if part_type == "image_url" or "image_url" in part:
                            image_field = part.get("image_url", "")
                            if isinstance(image_field, dict):
                                image_url = image_field.get("url", "")
                            elif isinstance(image_field, str):
                                image_url = image_field
                        else:
                            image_url = part.get("image", "")
                        # DashScope 仅接受 oss:// 或 http(s)://
                        if image_url.startswith(("oss://", "http://", "https://")):
                            ds_content.append({"image": image_url})
                        else:
                            log_warning(f"[FunctionCall] DashScope 不支持此图像格式，跳过")
                ds_messages.append({"role": role, "content": ds_content or [{"text": ""}]})
            else:
                ds_messages.append({"role": role, "content": content or ""})
        
        return ds_messages

    def _initial_messages(self, goal: str) -> List[Dict[str, Any]]:
        snapshot = self.world.snapshot()
        user_payload = {
            "goal": goal,
            "world": snapshot,
            "instructions": self._mission_text,
        }
        
        content_parts = [
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
            {"type": "text", "text": self._tool_instruction}
        ]
        
        # Add initial image
        img_base = self._get_current_image_base64()
        if img_base:
            content_parts.insert(0, {
                "type": "image_url",
                "image_url": {
                    "url": img_base
                }
            })
            
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content_parts},
        ]

    def _parse_arguments(self, arguments: Optional[str]) -> Dict[str, Any]:
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    def process_grasp_task(self, target_name: str, navigator, cam_name: str = "front") -> Dict[str, Any]:
        """
        Legacy entry point for single-target grasp tasks.
        Now delegates to the internal _run_fc_loop.
        """
        if navigator:
            self.set_navigator(navigator)
        elif self.navigator is None:
            raise ValueError("FunctionCallTaskProcessor 需要有效的导航控制器")
        self.observer.cam_name = cam_name
        
        # Initialize context
        self.world.set_goal(target_name)
        self._current_plan_id = uuid.uuid4().hex[:8]
        self._current_plan_entry = PlanContextEntry(
            plan_id=self._current_plan_id,
            goal=target_name,
            planned_steps=[],
        )
        self._timeline = []
        self.execution_history = []
        self._shared_runtime_extra = {}
        self._recovery_budget = {"total": 0, "per_code": {}, "start_time": time.time()}
        self.update_observation(None)
        self._publish_world_snapshot()
        self._failure_count = 0
        self._finalize_calls = 0

        messages = self._initial_messages(target_name)
        return self._run_fc_loop(target_name, messages)

    def process_long_horizon_task(self, instruction: str, navigator, cam_name: str = "front") -> Dict[str, Any]:
        """
        Entry point for long-horizon tasks.
        1. Decomposes instruction into subtasks.
        2. Executes each subtask sequentially via _run_fc_loop.
        """
        if navigator:
            self.set_navigator(navigator)
        elif self.navigator is None:
            raise ValueError("FunctionCallTaskProcessor 需要有效的导航控制器")
        self.observer.cam_name = cam_name
        
        # 1. Decompose
        log_info(f"🧠 [LongHorizon] Decomposing instruction: {instruction}")
        subtasks = self._decompose_task(instruction)
        if not subtasks:
            log_error("❌ [LongHorizon] Failed to decompose task.")
            return {"ok": False, "error": "decomposition_failed"}
        
        log_info(f"📋 [LongHorizon] Subtasks: {subtasks}")
        
        # Initialize shared context
        self._current_plan_id = uuid.uuid4().hex[:8]
        self._timeline = []
        self.execution_history = []
        self._shared_runtime_extra = {}
        self._recovery_budget = {"total": 0, "per_code": {}, "start_time": time.time()}
        self.update_observation(None)
        
        overall_trace = []
        
        # 2. Execute Subtasks
        for i, subtask in enumerate(subtasks):
            log_info(f"▶️ [LongHorizon] Starting Subtask {i+1}/{len(subtasks)}: {subtask}")
            
            # Update Goal
            self.world.set_goal(subtask)
            self._publish_world_snapshot()
            self._failure_count = 0
            self._finalize_calls = 0
            
            # Prepare messages for this subtask
            # We start fresh for each subtask to avoid context pollution, 
            # but we might want to carry over some world state (already in self.world).
            messages = self._initial_messages(subtask)
            
            # Run Loop
            result = self._run_fc_loop(subtask, messages)
            
            # Collect trace
            overall_trace.append({
                "subtask": subtask,
                "result": result
            })
            
            if not result.get("final_response", {}).get("content"):
                 # If loop didn't return a final response content (e.g. max rounds reached without success)
                 # We might consider it a failure.
                 pass
                 
            # Check for critical failure? 
            # For now, we continue to next subtask unless explicit stop?
            # Ideally, if a subtask fails, we should stop.
            # Let's check the last tool result or LLM response.
            # But _run_fc_loop returns a dict with 'final_response'.
            
        return {
            "ok": True,
            "instruction": instruction,
            "subtasks": subtasks,
            "trace": overall_trace
        }

    def _decompose_task(self, instruction: str) -> List[str]:
        """
        Uses LLM to decompose a high-level instruction into a list of subtasks.
        """
        prompt = f"""
        You are a robotic task planner.
        Decompose the following high-level instruction into a sequential list of short, actionable subtasks.
        Each subtask should be a simple sentence describing a single action (e.g., "Pick up the apple", "Place it in the box").
        
        Instruction: "{instruction}"
        
        Return ONLY a JSON list of strings. No markdown, no explanation.
        Example: ["Pick up the red block", "Place it on the green mat"]
        """
        
        messages = [{"role": "user", "content": prompt}]
        try:
            # Use a simple text call (no tools needed for decomposition)
            # We can reuse _call_llm but we need to temporarily disable tools or ignore them.
            # Or just use the client directly.
            if self.provider == "dashscope":
                 # DashScope doesn't support tool_choice="none" easily in our wrapper, 
                 # but we can just not pass tools.
                 # Let's use a simplified call.
                 response = dashscope.Generation.call(
                     model=self.model,
                     messages=[{"role": "user", "content": prompt}],
                     result_format="message"
                 )
                 if response.status_code == 200:
                     content = response.output.choices[0].message.content
                 else:
                     return []
            else:
                # Zhipu
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1
                )
                content = response.choices[0].message.content
                
            # Parse JSON
            # Clean up markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            subtasks = json.loads(content)
            if isinstance(subtasks, list):
                return subtasks
            return []
        except Exception as e:
            log_error(f"❌ [Decompose] Failed: {e}")
            return []

    def _run_fc_loop(self, target_name: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Internal loop for executing Function Calls for a single goal.
        """
        trace: List[Dict[str, Any]] = []
        final_result: Dict[str, Any] = {}

        for round_id in range(1, self.max_rounds + 1):
            log_info(f"[FunctionCall] 回合 {round_id}")
            message = self._call_llm(messages)
            messages.append(message)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # Handle multiple tool calls if present, though usually one
                for tool_call in tool_calls:
                    fn_name = tool_call.get("function", {}).get("name")
                    raw_args = tool_call.get("function", {}).get("arguments")
                    call_id = tool_call.get("id")
                    
                    args = self._parse_arguments(raw_args)
                    start_ts = time.time()
                    if fn_name == "finalize_target_pose":
                        self._finalize_calls += 1
                    result = self.tools.dispatch(fn_name, args)
                    if fn_name == "finalize_target_pose" and isinstance(result, dict):
                        result = dict(result)
                        result["finalize_attempt"] = self._finalize_calls
                    elapsed = time.time() - start_ts
                    
                    trace.append(
                        {
                            "round": round_id,
                            "function": fn_name,
                            "args": args,
                            "result": result,
                        }
                    )
                    status = "success" if result.get("ok") else "failure"
                    detail = result.get("reason") or result.get("error")
                    self._record_timeline_event(
                        stage="function_call",
                        node=fn_name or "unknown",
                        status=status,
                        detail=detail,
                        elapsed=elapsed,
                    )
                    self._append_execution_turn(
                        stage="function_call",
                        node=fn_name or "unknown",
                        status=status,
                        detail=json.dumps({"args": args, "result": result}, ensure_ascii=False),
                    )
                    
                    if status == "success":
                        log_success(f"[FunctionCall] {fn_name} 成功")
                        self._publish_world_snapshot()
                        self._failure_count = 0
                    else:
                        msg = f"[FunctionCall] {fn_name} 失败: {detail}"
                        log_warning(msg)
                        self._emit_ui_log(msg, "warning")
                        self._failure_count += 1
                        if self._failure_threshold > 0 and self._failure_count >= self._failure_threshold:
                            self._auto_recover(messages, trace, target_name)
                            self._failure_count = 0
                            
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    
                    # If finalize_target_pose has been called many times, inject a steering hint.
                    if fn_name == "finalize_target_pose" and self._finalize_calls >= 3:
                        messages.append({
                            "role": "user",
                            "content": (
                                "You have already aligned (finalize_target_pose) 3+ times. "
                                "Stop repeating finalize_target_pose; proceed to predict_grasp_point then execute_grasp."
                            ),
                        })
                    
                    if result.get("terminate"):
                        final_result = result
                        break
                
                if final_result:
                    break

                # Inject new image after tool execution for the next round
                img_base = self._get_current_image_base64()
                if img_base:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_base
                                }
                            },
                            {
                                "type": "text",
                                "text": "Current scene state after action."
                            }
                        ]
                    })
                
                continue

            # No tool call -> check if this is a valid final response or model hallucination
            content = message.get("content") or ""
            
            # Detect invalid/hallucinated responses (model outputting garbage instead of function calls)
            invalid_patterns = [
                "<|begin_of_box|>",
                "<|end_of_box|>",
                "```html",
                "```json",  # code block instead of function call
                "observe_scene\n",  # function name in text instead of call
                "predict_grasp_point\n",
                "execute_grasp\n",
                "finalize_target_pose\n",
            ]
            is_invalid_response = any(pat in content for pat in invalid_patterns)
            
            # Also check: if we haven't done key steps yet, this shouldn't be a final response
            has_grasp_result = any(
                t.get("function") in ("execute_grasp", "close_gripper") and t.get("result", {}).get("ok")
                for t in trace
            )
            
            # If response looks invalid or task isn't done, prompt model to retry with function call
            if is_invalid_response or (not has_grasp_result and "status" not in content.lower()):
                log_warning(f"[FunctionCall] 模型返回了无效响应，要求重试: {content[:100]}...")
                messages.append({
                    "role": "user",
                    "content": (
                        "ERROR: Your response was invalid. You MUST use function calls (tool_calls) to execute actions. "
                        "Do NOT output text descriptions of actions. "
                        "Please call the appropriate function now. What is the next function you need to call?"
                    )
                })
                continue
            
            try:
                final_result = json.loads(content)
            except Exception:
                # If can't parse as JSON and no grasp completed, treat as failure
                if not has_grasp_result:
                    final_result = {"status": "failure", "summary": f"Invalid model response: {content[:200]}"}
                else:
                    final_result = {"status": "success", "summary": content}
            self._record_timeline_event(
                stage="llm_response",
                node="final_response",
                status=str(final_result.get("status", "success")),
                detail=final_result.get("summary") or content,
            )
            self._append_execution_turn(
                stage="llm_response",
                node="final_response",
                status=str(final_result.get("status", "success")),
                detail=final_result.get("summary") or content,
            )
            break
        else:
            final_result = {"status": "failure", "summary": "max_rounds_exceeded"}

        success = str(final_result.get("status", "")).lower() == "success"
        if not success:
            log_warning(f"[FunctionCall] 任务失败: {final_result}")
            self._emit_ui_log(f"❌ 任务失败: {final_result.get('summary')}", "error")
        else:
            log_info(f"[FunctionCall] 任务完成: {final_result}")
            self._emit_ui_log(f"✅ 任务完成: {final_result.get('summary')}", "success")
        self._last_status_message = final_result.get("summary")
        self._finalize_plan_entry(success, final_result.get("summary") or final_result.get("reason"))
        return {
            "success": success,
            "reason": final_result.get("summary") or final_result.get("reason"),
            "final_response": final_result,
            "trace": trace,
            "timeline": self._timeline_payload(),
            "execution_history": [turn.__dict__ for turn in self.execution_history],
        }

    def _build_mission_instruction(self) -> str:
        text = textwrap.dedent(
            """
            Workflow guidance:

            General grounding rules:
            1) Always start with observe_scene(target, force_vlm=true).
            2) Any time the robot base moves (approach_far / finalize_target_pose / recover / navigation), you MUST call
            observe_scene(target, force_vlm=true) again before planning the next grasp.

            Locate:
            3) If the target is not visible, use rotate_scan and/or search_area, then observe_scene again. Repeat until visible
            or until scan attempts exceed a reasonable limit.

            Approach:
            4) When estimated_range > 2m, call approach_far, then observe_scene again; repeat until within range or limit reached.

            Align + Grasp (default: finalize once, then try grasp):
            5) Run finalize_target_pose ONCE (default behavior).
            6) After finalize_target_pose, MUST call observe_scene(target, force_vlm=true) again (camera view changed).
            7) Then run predict_grasp_point → execute_grasp → close_gripper.

            Failure handling (only on grasp execution failure):
            8) If execute_grasp OR close_gripper fails:
            - call recover(distance≈0.3)
            - observe_scene(target, force_vlm=true)
            - run finalize_target_pose ONCE as correction
            - observe_scene(target, force_vlm=true)
            - retry predict_grasp_point → execute_grasp → close_gripper
            - repeat this failure-retry loop up to grasp_attempts_max times (e.g., 3 total grasp attempts).
            If still failing after grasp_attempts_max, stop and return impossible.

            Handover:
            9) To hand the item to a user: navigate as needed, then handover_item → open_gripper.
            If navigation moved the base, re-observe before the handover step when appropriate.

            You may call observe_scene at any time when you think the situation has changed.
            Provide a final JSON summary {"status": "success|impossible", "summary": "..."} once done.
            """
        ).strip()
        return text

    def _build_tool_instruction(self) -> str:
        docs = textwrap.dedent(
            """
            Tool reference (units: meters for base poses, millimetres for TCP xyz, radians for angles):
            - observe_scene(target, phase, force_vlm, query): capture RGBD+VLM observation; optional query asks for textual analysis (adds "analysis" field).
            - rotate_scan(angle_deg) / search_area(turns, angle_deg): spin in place to look for target.
            - navigate_area(area/marker/pose) or navigate_to_pose(x,y,theta): move the base.
            - approach_far(target): coarse approach step when target is far away.
            - finalize_target_pose(): align the base using depth localization before grasping.
            - predict_grasp_point(): run ZeroGrasp; stores grasp pose inside runtime state.
            - execute_grasp(): execute the grasp pose computed previously.
            - open_gripper() / close_gripper().
            - handover_item(item): extend the item to a human after grasping.
            - get_object_state(object_id): returns the last known world model entry for that object.
            Always re-run observe_scene whenever the situation changes or after a recover.
            """
        ).strip()
        return docs

    def _finalize_plan_entry(self, success: bool, reason: Optional[str]) -> None:
        if not self._current_plan_entry:
            return
        entry = self._current_plan_entry
        entry.status = "completed" if success else "failed"
        entry.failure_reason = None if success else reason
        entry.executed = [{"node": turn.node, "status": turn.status} for turn in self.execution_history]
        entry.timestamp = time.time()
        self._current_plan_entry = None
        self._current_plan_id = None

    def _execute_internal_tool(
        self,
        fn_name: str,
        args: Dict[str, Any],
        trace: List[Dict[str, Any]],
        stage: str,
    ) -> Dict[str, Any]:
        start_ts = time.time()
        result = self.tools.dispatch(fn_name, args)
        elapsed = time.time() - start_ts
        trace.append(
            {
                "round": "auto",
                "function": fn_name,
                "args": args,
                "result": result,
                "stage": stage,
            }
        )
        status = "success" if result.get("ok") else "failure"
        detail = result.get("reason") or result.get("error")
        self._record_timeline_event(stage=stage, node=fn_name, status=status, detail=detail, elapsed=elapsed)
        self._append_execution_turn(
            stage=stage,
            node=fn_name,
            status=status,
            detail=json.dumps({"args": args, "result": result}, ensure_ascii=False),
        )
        if status == "success":
            self._publish_world_snapshot()
        else:
            self._emit_ui_log(f"[Auto] {fn_name} 失败: {detail}", "warning")
        return result

    def _auto_recover(self, messages: List[Dict[str, Any]], trace: List[Dict[str, Any]], target: str) -> None:
        log_warning("[FunctionCall] 连续失败，自动触发 recover + observe")
        recover_result = self._execute_internal_tool(
            "recover",
            {"distance": 0.3},
            trace,
            stage="auto_recover",
        )
        observe_result: Optional[Dict[str, Any]] = None
        if target:
            observe_result = self._execute_internal_tool(
                "observe_scene",
                {"target": target, "force_vlm": True},
                trace,
                stage="auto_observe",
            )
        summary = {
            "auto_recover": recover_result,
            "auto_observe": observe_result,
        }
        messages.append(
            {
                "role": "system",
                "content": json.dumps(
                    {
                        "auto_recover": "recover + observe executed",
                        "details": summary,
                        "hint": "Please continue planning with the updated observation.",
                    },
                    ensure_ascii=False,
                ),
            }
        )
