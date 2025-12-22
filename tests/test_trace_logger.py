import unittest
import os
import shutil
import json
from unittest.mock import MagicMock
from voice.control.executor import SkillExecutor
from voice.control.task_structures import PlanNode, SkillRuntime

class TestTraceLogger(unittest.TestCase):
    def setUp(self):
        # Setup environment for testing
        self.test_log_root = "test_runs"
        if os.path.exists(self.test_log_root):
            shutil.rmtree(self.test_log_root)
            
        # Mock TraceLogger to use test directory
        # We need to patch the TraceLogger inside executor, but since it's instantiated in __init__,
        # we can just modify the instance after creation or patch the class before.
        # Here we will just let it run and check the default "runs" directory or modify the instance.
        
        self.executor = SkillExecutor()
        # Redirect logger to test dir
        self.executor.trace_logger.log_root = self.test_log_root
        # Re-init logger with new root
        self.executor.trace_logger.__init__(log_root=self.test_log_root)
        
        self.runtime = SkillRuntime(navigator=MagicMock())

    def tearDown(self):
        if os.path.exists(self.test_log_root):
            shutil.rmtree(self.test_log_root)

    def test_trace_logging(self):
        # 1. Execute a simple skill (open_gripper)
        node = PlanNode(type="action", name="open_gripper", args={})
        # Mock the handler to avoid hardware calls
        self.executor._skill_open_gripper = MagicMock(return_value=MagicMock(status="success", reason=None, evidence={"mock": True}))
        
        result = self.executor.execute(node, self.runtime)
        
        # 2. Verify execution success
        self.assertEqual(result.status, "success")
        
        # 3. Verify log file creation
        log_dir = self.executor.trace_logger.log_dir
        log_file = self.executor.trace_logger.log_file
        
        self.assertTrue(os.path.exists(log_dir))
        self.assertTrue(os.path.exists(log_file))
        
        # 4. Verify log content
        with open(log_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            
            self.assertEqual(record["skill_name"], "open_gripper")
            self.assertEqual(record["exec_status"], "success")
            self.assertEqual(record["step_id"], 1)
            self.assertIn("timestamp", record)
            self.assertIn("episode_id", record)

if __name__ == "__main__":
    unittest.main()
