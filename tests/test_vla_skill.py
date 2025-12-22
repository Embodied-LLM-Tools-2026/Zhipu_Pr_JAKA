import unittest
from unittest.mock import MagicMock
from voice.control.executor import SkillExecutor, SkillRuntime, PlanNode, ExecutionResult, FailureCode

class TestVLASkill(unittest.TestCase):
    def setUp(self):
        self.executor = SkillExecutor(node=MagicMock(), navigator=MagicMock())
        self.executor.trace_logger = MagicMock()
        self.executor._ensure_arm_client = MagicMock(return_value=MagicMock())
        
    def test_vla_grasp_finish_success(self):
        # Mock observation
        obs = MagicMock()
        obs.found = True
        
        node = PlanNode(type="action", name="vla_grasp_finish", args={"instruction": "pick up the apple"})
        runtime = SkillRuntime(navigator=MagicMock(), world_model=None, observation=obs, extra={})
        
        # Force random to return success (> 0.10)
        import random
        with unittest.mock.patch('random.random', return_value=0.5):
            result = self.executor.execute(node, runtime)
            
        self.assertEqual(result.status, "success")
        self.assertTrue(result.verified) # Should be verified by _verify_grasp_success
        self.assertEqual(result.evidence["model"], "vla-large-v1")
        
    def test_vla_grasp_finish_failure(self):
        # Mock observation
        obs = MagicMock()
        obs.found = True
        
        node = PlanNode(type="action", name="vla_grasp_finish", args={})
        runtime = SkillRuntime(navigator=MagicMock(), world_model=None, observation=obs, extra={})
        
        # Force random to return failure (< 0.05 -> OOB)
        import random
        with unittest.mock.patch('random.random', return_value=0.01):
            result = self.executor.execute(node, runtime)
            
        self.assertEqual(result.status, "failure")
        self.assertEqual(result.failure_code, FailureCode.VLA_POLICY_OOB)

if __name__ == '__main__':
    unittest.main()
