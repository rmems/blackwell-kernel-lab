#!/usr/bin/env python3
"""Exercise the GPU-validation check publisher without calling GitHub."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from host_cli import run_python_script

PUBLISHER = Path(__file__).with_name("publish_gpu_validation.py")
REPOSITORY = "rmems/blackwell-kernel-lab"
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


def pr_target_event() -> dict:
    return {
        "repository": {"full_name": REPOSITORY},
        "pull_request": {
            "base": {"sha": BASE_SHA},
            "head": {"sha": HEAD_SHA, "repo": {"full_name": REPOSITORY}},
        },
    }


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.event_path = self.root / "event.json"

    def run_dry_run(self, validation_result: str):
        self.event_path.write_text(json.dumps(pr_target_event()))
        environment = dict(
            os.environ,
            GITHUB_EVENT_PATH=str(self.event_path),
            GITHUB_EVENT_NAME="pull_request_target",
            GITHUB_REPOSITORY=REPOSITORY,
            GITHUB_RUN_ID="1234",
            VALIDATION_RESULT=validation_result,
        )
        return run_python_script(
            PUBLISHER, "--dry-run",
            check=False,
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
        )

    def test_pr_target_dry_run_publishes_gpu_validation_on_pr_head(self):
        """A trusted target workflow must satisfy the PR head, not its base SHA."""
        result = self.run_dry_run("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["name"], "GPU validation")
        self.assertEqual(payload["head_sha"], HEAD_SHA)
        self.assertEqual(payload["conclusion"], "success")

    def test_failed_validation_publishes_a_failing_required_check(self):
        """A failed trusted validation must block the PR head instead of skipping."""
        result = self.run_dry_run("failure")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["head_sha"], HEAD_SHA)
        self.assertEqual(payload["conclusion"], "failure")


if __name__ == "__main__":
    unittest.main()
