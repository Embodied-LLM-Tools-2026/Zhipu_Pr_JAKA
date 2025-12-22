import unittest
from unittest.mock import MagicMock, patch
from voice.control.executor import SkillExecutor, SkillRuntime, PlanNode, ExecutionResult, FailureCode
from voice.control.recovery import RecoveryPlan

class TestReflexRecovery(unittest.TestCase):
    def setUp(self):
        self.executor = SkillExecutor(node=MagicMock(), navigator=MagicMock())
        self.executor.trace_logger = MagicMock()
        self.executor._ensure_arm_client = MagicMock(return_value=MagicMock())
        
    def test_ik_fail_reflex(self):
        # Scenario: IK_FAIL -> Nudge -> Retry Success
        
        fail_result = ExecutionResult(status="failure", failure_code=FailureCode.IK_FAIL, reason="ik_fail")
        rec_success = ExecutionResult(status="success")
        retry_success = ExecutionResult(status="success")
        
        def side_effect(node, runtime):
            if node.name == "execute_grasp":
                if runtime.extra.get("is_retry"):
                    return retry_success
                return fail_result
            elif node.name == "recover":
                # Verify args for nudge
                if node.args.get("mode") == "nudge_base":
                    return rec_success
            return ExecutionResult(status="failure", reason="unknown")
            
        self.executor._execute_single_shot = MagicMock(side_effect=side_effect)
        
        node = PlanNode(type="action", name="execute_grasp", args={})
        runtime = SkillRuntime(navigator=MagicMock(), world_model=None, observation=None, extra={})
        
        final_result = self.executor.execute(node, runtime)
        
        self.assertEqual(final_result.status, "success")
        
        # Verify calls
        calls = self.executor._execute_single_shot.call_args_list
        self.assertEqual(len(calls), 3)
        
        # 1. Original Fail
        self.assertEqual(calls[0][0][0].name, "execute_grasp")
        
        # 2. Recovery (Nudge)
        self.assertEqual(calls[1][0][0].name, "recover")
        self.assertEqual(calls[1][0][0].args["mode"], "nudge_base")
        
        # 3. Retry Success
        self.assertEqual(calls[2][0][0].name, "execute_grasp")
        self.assertTrue(calls[2][0][1].extra.get("is_retry"))

    def test_grasp_fail_reflex(self):
        # Scenario: GRASP_FAIL -> Open -> Reset -> Retry Success
        
        fail_result = ExecutionResult(status="failure", failure_code=FailureCode.GRASP_FAIL, reason="grasp_fail")
        op_success = ExecutionResult(status="success")
        reset_success = ExecutionResult(status="success")
        retry_success = ExecutionResult(status="success")
        
        def side_effect(node, runtime):
            if node.name == "execute_grasp":
                if runtime.extra.get("is_retry"):
                    return retry_success
                return fail_result
            elif node.name == "open_gripper":
                return op_success
            elif node.name == "recover":
                if node.args.get("mode") == "reset_arm":
                    return reset_success
            return ExecutionResult(status="failure", reason=f"unknown {node.name}")
            
        self.executor._execute_single_shot = MagicMock(side_effect=side_effect)
        
        node = PlanNode(type="action", name="execute_grasp", args={})
        runtime = SkillRuntime(navigator=MagicMock(), world_model=None, observation=None, extra={})
        
        final_result = self.executor.execute(node, runtime)
        
        self.assertEqual(final_result.status, "success")
        
        # Verify calls: Original -> Open -> Reset -> Retry
        calls = self.executor._execute_single_shot.call_args_list
        self.assertEqual(len(calls), 4)
        
        self.assertEqual(calls[0][0][0].name, "execute_grasp")
        self.assertEqual(calls[1][0][0].name, "open_gripper")
        self.assertEqual(calls[2][0][0].name, "recover")
        self.assertEqual(calls[2][0][0].args["mode"], "reset_arm")
        self.assertEqual(calls[3][0][0].name, "execute_grasp")

if __name__ == '__main__':
    unittest.main()
