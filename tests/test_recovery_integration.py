import unittest
from unittest.mock import MagicMock, patch
from voice.control.executor import SkillExecutor, SkillRuntime, PlanNode, ExecutionResult, FailureCode
from voice.control.recovery import RecoveryPlan

class TestRecoveryIntegration(unittest.TestCase):
    def setUp(self):
        # Mock ROS node
        self.executor = SkillExecutor(node=MagicMock(), navigator=MagicMock())
        # Mock trace logger
        self.executor.trace_logger = MagicMock()
        
    def test_recovery_l1_retry(self):
        # Scenario: Skill fails with NAV_BLOCKED -> Recovery (backoff) -> Retry -> Success
        
        fail_result = ExecutionResult(status="failure", failure_code=FailureCode.NAV_BLOCKED, reason="blocked")
        rec_success = ExecutionResult(status="success")
        retry_success = ExecutionResult(status="success")
        
        def side_effect(node, runtime):
            if node.name == "navigate_to":
                if runtime.extra.get("is_retry"):
                    return retry_success
                return fail_result
            elif node.name == "recover":
                return rec_success
            return ExecutionResult(status="failure", reason="unknown")
            
        # We mock the internal method _execute_single_shot
        self.executor._execute_single_shot = MagicMock(side_effect=side_effect)
        
        node = PlanNode(type="action", name="navigate_to", args={})
        runtime = SkillRuntime(navigator=None, world_model=None, observation=None, extra={})
        
        final_result = self.executor.execute(node, runtime)
        
        self.assertEqual(final_result.status, "success")
        
        # Verify calls: 1. Original, 2. Recovery, 3. Retry
        self.assertEqual(self.executor._execute_single_shot.call_count, 3)
        
        # Check arguments of calls
        calls = self.executor._execute_single_shot.call_args_list
        
        # 1. Original
        self.assertEqual(calls[0][0][0].name, "navigate_to")
        self.assertFalse(calls[0][0][1].extra.get("is_retry"))
        
        # 2. Recovery
        self.assertEqual(calls[1][0][0].name, "recover")
        self.assertEqual(calls[1][0][1].extra.get("recovery_level"), "L1")
        
        # 3. Retry
        self.assertEqual(calls[2][0][0].name, "navigate_to")
        self.assertTrue(calls[2][0][1].extra.get("is_retry"))

if __name__ == '__main__':
    unittest.main()
