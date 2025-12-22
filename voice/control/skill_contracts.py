from typing import Dict, List, Any

SKILL_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "search_area": {
        "preconditions": ["Robot localized", "Navigation stack active"],
        "invariants": ["Safety collision avoidance active"],
        "failure_codes": ["NO_OBSERVATION", "NAV_BLOCKED"],
        "description": "Rotates and scans the environment to find a target object."
    },
    "approach_far": {
        "preconditions": ["Target detected in observation", "Distance > 1.0m"],
        "invariants": ["Keep target in field of view"],
        "failure_codes": ["NAV_BLOCKED", "LOST_TARGET"],
        "description": "Navigates to a close proximity of the target."
    },
    "finalize_target_pose": {
        "preconditions": ["Target within 1.0m", "RGBD stream valid"],
        "invariants": ["Robot base stationary"],
        "failure_codes": ["NO_OBSERVATION", "VERIFICATION_FAILED"],
        "description": "Aligns the robot and computes the precise target pose for grasping."
    },
    "predict_grasp_point": {
        "preconditions": ["Target pose finalized", "ZeroGrasp service available"],
        "invariants": ["Camera calibration valid"],
        "failure_codes": ["ZEROGRASP_FAILED", "NO_OBSERVATION"],
        "description": "Computes 6-DOF grasp poses using ZeroGrasp model."
    },
    "execute_grasp": {
        "preconditions": ["Valid grasp pose", "Arm ready"],
        "invariants": ["Collision free path"],
        "failure_codes": ["IK_FAIL", "GRASP_FAIL", "GRASP_SLIP"],
        "description": "Executes the computed grasp trajectory."
    },
    "vla_execute": {
        "preconditions": ["Target visible", "VLA model loaded"],
        "invariants": ["Visual servoing active"],
        "failure_codes": ["VLA_NO_EFFECT", "NO_OBSERVATION"],
        "description": "Uses VLA model to execute instructions (grasp, place, etc)."
    },
    "rotate_scan": {
        "preconditions": ["Base motors active"],
        "invariants": ["Stay within workspace bounds"],
        "failure_codes": ["ROTATE_FAILED"],
        "description": "Performs an in-place rotation to scan for targets."
    },
    "navigate_area": {
        "preconditions": ["Map available", "Goal reachable"],
        "invariants": ["Avoid dynamic obstacles"],
        "failure_codes": ["NAV_BLOCKED", "NAVIGATOR_UNAVAILABLE"],
        "description": "Navigates to a specific area in the map."
    },
    "open_gripper": {
        "preconditions": ["Gripper connected"],
        "invariants": [],
        "failure_codes": ["GRIPPER_UNAVAILABLE"],
        "description": "Opens the robot gripper."
    },
    "close_gripper": {
        "preconditions": ["Gripper connected"],
        "invariants": [],
        "failure_codes": ["GRIPPER_UNAVAILABLE"],
        "description": "Closes the robot gripper."
    },
    "recover": {
        "preconditions": ["None"],
        "invariants": ["Safety limits"],
        "failure_codes": ["RECOVERY_FAILED"],
        "description": "Executes recovery actions like backoff or reset."
    },
    "pick": {
        "preconditions": ["Target reachable", "Gripper empty"],
        "invariants": ["Collision free"],
        "failure_codes": ["GRASP_FAIL", "IK_FAIL"],
        "description": "High-level pick skill (composite)."
    },
    "place": {
        "preconditions": ["Holding object", "Place target reachable"],
        "invariants": ["Collision free"],
        "failure_codes": ["IK_FAIL", "GRIPPER_UNAVAILABLE"],
        "description": "High-level place skill."
    },
    "return_home": {
        "preconditions": ["None"],
        "invariants": ["Safety limits"],
        "failure_codes": ["IK_FAIL"],
        "description": "Moves the robot arm to the home position."
    },
    "handover_item": {
        "preconditions": ["Holding object", "Human detected"],
        "invariants": ["Safety limits"],
        "failure_codes": ["NO_OBSERVATION", "IK_FAIL"],
        "description": "Hands over the object to a human."
    },
    "align_tcp_to_target": {
        "preconditions": ["Target visible in depth", "Arm ready"],
        "invariants": ["Collision free path"],
        "failure_codes": ["NO_OBSERVATION", "IK_FAIL", "ARM_UNAVAILABLE"],
        "description": "Moves the TCP to a point offset from the target, aligned with the robot-target line."
    }
}
