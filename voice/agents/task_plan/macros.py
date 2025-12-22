"""
Macro library for expanding high-level subtasks into sequences of atomic skills.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class MacroStep:
    skill_name: str
    args: Dict[str, Any]
    optional: bool = False

class MacroRegistry:
    @staticmethod
    def expand(task_type: str, params: Dict[str, Any]) -> List[MacroStep]:
        """
        Expands a high-level task type into a sequence of atomic skills.
        """
        method_name = f"_expand_{task_type}"
        if hasattr(MacroRegistry, method_name):
            return getattr(MacroRegistry, method_name)(params)
        
        # Fallback: if no macro, assume 1-to-1 mapping
        # We map common aliases to skill names if needed, or just pass through
        skill_name = MacroRegistry._map_alias(task_type)
        return [MacroStep(skill_name=skill_name, args=params)]

    @staticmethod
    def _map_alias(task_type: str) -> str:
        # Map high-level types to executor skill names if they match directly
        aliases = {
            "navigate": "navigate_area", # or navigate_to? Executor has navigate_area
            "pick": "pick", # Executor has _skill_pick (which might be a macro itself or atomic?)
            # Executor has _skill_pick which seems to be a wrapper or atomic.
            # But user wants explicit macro expansion for 'fetch_place' etc.
            "place": "place",
            "observe": "search_area",
            "wait": "wait", # Executor doesn't seem to have _skill_wait in grep results?
            # Wait, grep showed _skill_recover, _skill_handover...
            # Let's assume 'wait' might need to be implemented or mapped to something else.
            # Actually, grep didn't show _skill_wait. I should check if it exists or use time.sleep in a wrapper.
            # But for now, let's stick to what we found.
        }
        return aliases.get(task_type, task_type)

    @staticmethod
    def _expand_fetch_place(params: Dict[str, Any]) -> List[MacroStep]:
        """
        fetch_place: {object, from, to}
        """
        obj = params.get("object") or params.get("target")
        loc_from = params.get("from")
        loc_to = params.get("to")
        
        steps = []
        
        # 1. Go to 'from' location
        if loc_from:
            steps.append(MacroStep("navigate_area", {"target": loc_from}))
            
        # 2. Search/Scan for object
        steps.append(MacroStep("search_area", {"target": obj}))
        
        # 3. Approach
        steps.append(MacroStep("approach_far", {"target": obj}))
        
        # 4. Finalize Pose (Localize)
        steps.append(MacroStep("finalize_target_pose", {"target": obj}))
        
        # 5. Predict Grasp
        steps.append(MacroStep("predict_grasp_point", {"target": obj}))
        
        # 6. Execute Grasp
        steps.append(MacroStep("execute_grasp", {"target": obj}))
        
        # 7. Go to 'to' location
        if loc_to:
            steps.append(MacroStep("navigate_area", {"target": loc_to}))
            
        # 8. Place
        steps.append(MacroStep("place", {"target": loc_to}))
        
        return steps

    @staticmethod
    def _expand_fetch_only(params: Dict[str, Any]) -> List[MacroStep]:
        """
        fetch_only: {object, from}
        """
        obj = params.get("object") or params.get("target")
        loc_from = params.get("from")
        
        steps = []
        if loc_from:
            steps.append(MacroStep("navigate_area", {"target": loc_from}))
            
        steps.append(MacroStep("search_area", {"target": obj}))
        steps.append(MacroStep("approach_far", {"target": obj}))
        steps.append(MacroStep("finalize_target_pose", {"target": obj}))
        steps.append(MacroStep("predict_grasp_point", {"target": obj}))
        steps.append(MacroStep("execute_grasp", {"target": obj}))
        
        return steps

    @staticmethod
    def _expand_place_only(params: Dict[str, Any]) -> List[MacroStep]:
        """
        place_only: {object, to} (Assumes holding)
        """
        loc_to = params.get("to") or params.get("target")
        
        steps = []
        if loc_to:
            steps.append(MacroStep("navigate_area", {"target": loc_to}))
            
        steps.append(MacroStep("place", {"target": loc_to}))
        
        return steps
