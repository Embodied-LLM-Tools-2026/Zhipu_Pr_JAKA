import unittest
from voice.agents.task_plan.schema import validate_plan, PlanValidationError, Plan

class TestPlanValidation(unittest.TestCase):
    def test_valid_plan(self):
        plan_dict = {
            "goal": "Clean the table",
            "plan_id": "plan_001",
            "subtasks": [
                {
                    "id": "1",
                    "type": "navigate",
                    "params": {"target": "table"},
                    "timeout_s": 30.0
                },
                {
                    "id": "2",
                    "type": "pick",
                    "params": {"target": "cup"},
                    "depends_on": ["1"]
                }
            ]
        }
        plan = validate_plan(plan_dict)
        self.assertIsInstance(plan, Plan)
        self.assertEqual(len(plan.subtasks), 2)
        self.assertEqual(plan.subtasks[1].depends_on, ["1"])

    def test_duplicate_ids(self):
        plan_dict = {
            "goal": "Fail",
            "plan_id": "fail_001",
            "subtasks": [
                {"id": "1", "type": "wait", "params": {"duration_s": 1}},
                {"id": "1", "type": "wait", "params": {"duration_s": 1}}
            ]
        }
        with self.assertRaises(PlanValidationError) as cm:
            validate_plan(plan_dict)
        self.assertIn("Duplicate subtask IDs", str(cm.exception))

    def test_missing_dependency(self):
        plan_dict = {
            "goal": "Fail",
            "plan_id": "fail_002",
            "subtasks": [
                {"id": "1", "type": "wait", "params": {"duration_s": 1}, "depends_on": ["99"]}
            ]
        }
        with self.assertRaises(PlanValidationError) as cm:
            validate_plan(plan_dict)
        self.assertIn("unknown ID '99'", str(cm.exception))

    def test_cycle_detection(self):
        plan_dict = {
            "goal": "Fail",
            "plan_id": "fail_003",
            "subtasks": [
                {"id": "1", "type": "wait", "params": {"duration_s": 1}, "depends_on": ["2"]},
                {"id": "2", "type": "wait", "params": {"duration_s": 1}, "depends_on": ["1"]}
            ]
        }
        with self.assertRaises(PlanValidationError) as cm:
            validate_plan(plan_dict)
        self.assertIn("Dependency cycle detected", str(cm.exception))

    def test_invalid_type(self):
        plan_dict = {
            "goal": "Fail",
            "plan_id": "fail_004",
            "subtasks": [
                {"id": "1", "type": "dance", "params": {}}
            ]
        }
        with self.assertRaises(PlanValidationError) as cm:
            validate_plan(plan_dict)
        self.assertIn("invalid type 'dance'", str(cm.exception))

    def test_missing_params(self):
        plan_dict = {
            "goal": "Fail",
            "plan_id": "fail_005",
            "subtasks": [
                {"id": "1", "type": "navigate", "params": {}} # Missing target
            ]
        }
        with self.assertRaises(PlanValidationError) as cm:
            validate_plan(plan_dict)
        self.assertIn("missing required params", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
