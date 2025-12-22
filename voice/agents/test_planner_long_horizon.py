import unittest
from unittest.mock import MagicMock, patch
from voice.agents.planner import BehaviorPlanner
from voice.agents.task_plan.schema import Plan, Subtask

class TestPlannerLongHorizon(unittest.TestCase):
    def setUp(self):
        self.planner = BehaviorPlanner(llm_api_key="test_key")
        self.mock_world = MagicMock()
        self.mock_world.snapshot.return_value = {"areas": {}, "objects": {}}

    @patch("voice.agents.planner.requests.post")
    def test_make_long_horizon_plan_success(self, mock_post):
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": """
                    {
                        "goal": "Test Goal",
                        "plan_id": "test_01",
                        "subtasks": [
                            {"id": "1", "type": "navigate", "params": {"target": "kitchen"}}
                        ]
                    }
                    """
                }
            }]
        }
        mock_post.return_value = mock_response
        
        plan = self.planner.make_long_horizon_plan("Test Goal", self.mock_world)
        
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.goal, "Test Goal")
        self.assertEqual(len(plan.subtasks), 1)
        self.assertEqual(plan.subtasks[0].type, "navigate")

    @patch("voice.agents.planner.requests.post")
    def test_make_long_horizon_plan_retry_on_validation_error(self, mock_post):
        # First response: Invalid (missing plan_id)
        # Second response: Valid
        
        bad_response = {
            "choices": [{
                "message": {
                    "content": """
                    {
                        "goal": "Test Goal",
                        "subtasks": []
                    }
                    """
                }
            }]
        }
        
        good_response = {
            "choices": [{
                "message": {
                    "content": """
                    {
                        "goal": "Test Goal",
                        "plan_id": "retry_01",
                        "subtasks": [
                            {"id": "1", "type": "wait", "params": {"duration_s": 1}}
                        ]
                    }
                    """
                }
            }]
        }
        
        mock_post.side_effect = [
            MagicMock(json=MagicMock(return_value=bad_response)),
            MagicMock(json=MagicMock(return_value=good_response))
        ]
        
        plan = self.planner.make_long_horizon_plan("Test Goal", self.mock_world)
        
        self.assertEqual(plan.plan_id, "retry_01")
        self.assertEqual(mock_post.call_count, 2)
        
        # Check if feedback was sent in 2nd call
        second_call_args = mock_post.call_args_list[1]
        payload = second_call_args[1]['json']
        user_msg = json.loads(payload['messages'][1]['content'])
        self.assertIn("feedback_from_previous_attempt", user_msg)
        self.assertIn("plan_id", user_msg["feedback_from_previous_attempt"]) # Error message should mention missing plan_id

if __name__ == "__main__":
    unittest.main()
