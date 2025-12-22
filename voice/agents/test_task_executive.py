import unittest
from unittest.mock import MagicMock, ANY
from voice.agents.task_executive import TaskExecutive, SubtaskState
from voice.agents.task_plan.schema import Plan, Subtask, RetryPolicy
from voice.control.task_structures import ExecutionResult, FailureCode
from voice.control.executor import SkillRuntime

class TestTaskExecutive(unittest.TestCase):
    def setUp(self):
        self.mock_executor = MagicMock()
        self.mock_world = MagicMock()
        self.mock_recovery = MagicMock()
        self.executive = TaskExecutive(
            executor=self.mock_executor,
            world_model=self.mock_world,
            recovery_manager=self.mock_recovery
        )
        self.runtime = SkillRuntime(navigator=None)

    def test_run_plan_success(self):
        # Setup Plan
        plan = Plan(
            goal="Test Goal",
            plan_id="test_plan",
            subtasks=[
                Subtask(id="1", type="navigate", params={"target": "kitchen"}),
                Subtask(id="2", type="pick", params={"target": "cup"}, depends_on=["1"])
            ]
        )
        
        # Setup Executor Success
        self.mock_executor.execute.return_value = ExecutionResult(status="success", node="test")
        
        # Run
        result = self.executive.run_plan(plan, self.runtime)
        
        # Verify
        self.assertTrue(result.success)
        self.assertEqual(result.final_state["1"], SubtaskState.DONE)
        self.assertEqual(result.final_state["2"], SubtaskState.DONE)
        self.assertEqual(self.mock_executor.execute.call_count, 2)

    def test_run_plan_dependency_skip(self):
        # Setup Plan
        plan = Plan(
            goal="Test Goal",
            plan_id="test_plan",
            subtasks=[
                Subtask(id="1", type="navigate", params={"target": "kitchen"}),
                Subtask(id="2", type="pick", params={"target": "cup"}, depends_on=["1"])
            ]
        )
        
        # Setup Executor Failure for Task 1
        self.mock_executor.execute.side_effect = [
            ExecutionResult(status="failure", node="navigate"), # Task 1 fails
        ]
        
        # Run
        result = self.executive.run_plan(plan, self.runtime)
        
        # Verify
        self.assertFalse(result.success)
        self.assertEqual(result.final_state["1"], SubtaskState.FAILED)
        self.assertEqual(result.final_state["2"], SubtaskState.SKIPPED)

    def test_done_if_precheck(self):
        # Setup Plan
        plan = Plan(
            goal="Test Goal",
            plan_id="test_plan",
            subtasks=[
                Subtask(id="1", type="navigate", params={"target": "kitchen"}, done_if="at(kitchen)")
            ]
        )
        
        # Setup World Model to satisfy done_if
        self.mock_world.areas = {"kitchen": MagicMock(pose=[1, 1, 0])}
        self.mock_world.robot = {"pose": [1, 1, 0]}
        
        # Run
        result = self.executive.run_plan(plan, self.runtime)
        
        # Verify
        self.assertTrue(result.success)
        self.assertEqual(result.final_state["1"], SubtaskState.DONE)
        self.mock_executor.execute.assert_not_called() # Should skip execution

if __name__ == "__main__":
    unittest.main()
