import unittest
from unittest.mock import MagicMock, call
from voice.agents.task_executive import TaskExecutive, SubtaskState
from voice.agents.task_plan.schema import Plan, Subtask
from voice.control.task_structures import ExecutionResult
from voice.control.executor import SkillRuntime

class TestTaskExecutiveMacros(unittest.TestCase):
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

    def test_fetch_place_macro_expansion(self):
        # Setup Plan with fetch_place
        plan = Plan(
            goal="Fetch cup",
            plan_id="p1",
            subtasks=[
                Subtask(
                    id="1", 
                    type="fetch_place", 
                    params={"object": "cup", "from": "kitchen", "to": "table"}
                )
            ]
        )
        
        # Setup Executor to succeed all steps
        self.mock_executor.execute.return_value = ExecutionResult(status="success", node="test")
        
        # Run
        result = self.executive.run_plan(plan, self.runtime)
        
        # Verify Success
        self.assertTrue(result.success)
        self.assertEqual(result.final_state["1"], SubtaskState.DONE)
        
        # Verify Sequence of Calls
        # Expected steps: navigate_area(kitchen), search_area(cup), approach_far(cup), 
        # finalize_target_pose(cup), predict_grasp_point(cup), execute_grasp(cup), 
        # navigate_area(table), place(table)
        
        calls = self.mock_executor.execute.call_args_list
        self.assertEqual(len(calls), 8)
        
        # Check first call (navigate to kitchen)
        node1 = calls[0][0][0]
        self.assertEqual(node1.name, "navigate_area")
        self.assertEqual(node1.args["target"], "kitchen")
        
        # Check last call (place on table)
        node8 = calls[7][0][0]
        self.assertEqual(node8.name, "place")
        self.assertEqual(node8.args["target"], "table")

    def test_macro_step_failure(self):
        # Setup Plan
        plan = Plan(
            goal="Fetch cup",
            plan_id="p2",
            subtasks=[
                Subtask(id="1", type="fetch_place", params={"object": "cup", "from": "kitchen", "to": "table"})
            ]
        )
        
        # Fail on the 2nd step (search_area)
        self.mock_executor.execute.side_effect = [
            ExecutionResult(status="success", node="navigate_area"),
            ExecutionResult(status="failure", node="search_area"),
        ]
        
        # Run
        result = self.executive.run_plan(plan, self.runtime)
        
        # Verify Failure
        self.assertFalse(result.success)
        self.assertEqual(result.final_state["1"], SubtaskState.FAILED)
        self.assertEqual(self.mock_executor.execute.call_count, 2)

if __name__ == "__main__":
    unittest.main()
