#!/usr/bin/env python3
"""Regression tests for the text-only GitHub Actions policy checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


CHECKER_PATH = Path(__file__).with_name("check_gpu_ci_policy.py")
SPEC = importlib.util.spec_from_file_location("gpu_ci_policy", CHECKER_PATH)
assert SPEC and SPEC.loader
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


class PolicyTests(unittest.TestCase):
    def test_gpu_workflow_rejects_continue_on_error(self):
        workflow = POLICY.load(POLICY.GPU_YML) + "\ncontinue-on-error: true\n"
        with self.assertRaises(POLICY.PolicyError):
            POLICY.check_gpu_smoke_workflow(workflow)


if __name__ == "__main__":
    unittest.main()
