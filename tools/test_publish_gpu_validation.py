#!/usr/bin/env python3
"""Exercise the GPU-validation check publisher without calling GitHub."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PUBLISHER = Path(__file__).with_name("publish_gpu_validation.py")
REPOSITORY = "rmems/blackwell-kernel-lab"
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


class PublisherTests(unittest.TestCase):
    def test_pr_target_dry_run_publishes_gpu_validation_on_pr_head(self):
        """A trusted target workflow must satisfy the PR head, not its base SHA."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event = root / "event.json"
            event.write_text(json.dumps({
                "repository": {"full_name": REPOSITORY},
                "pull_request": {
                    "base": {"sha": BASE_SHA},
                    "head": {"sha": HEAD_SHA, "repo": {"full_name": REPOSITORY}},
                },
            }))
            environment = dict(os.environ, GITHUB_EVENT_PATH=str(event),
                               GITHUB_EVENT_NAME="pull_request_target",
                               GITHUB_REPOSITORY=REPOSITORY, GITHUB_RUN_ID="1234",
                               VALIDATION_RESULT="success")
            result = subprocess.run(
                ["python3", str(PUBLISHER), "--dry-run"], cwd=root,
                env=environment, capture_output=True, text=True, check=False,
            )  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["name"], "GPU validation")
        self.assertEqual(payload["head_sha"], HEAD_SHA)
        self.assertEqual(payload["conclusion"], "success")

    def test_failed_validation_publishes_a_failing_required_check(self):
        """A failed trusted validation must block the PR head instead of skipping."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event = root / "event.json"
            event.write_text(json.dumps({
                "repository": {"full_name": REPOSITORY},
                "pull_request": {
                    "base": {"sha": BASE_SHA},
                    "head": {"sha": HEAD_SHA, "repo": {"full_name": REPOSITORY}},
                },
            }))
            environment = dict(os.environ, GITHUB_EVENT_PATH=str(event),
                               GITHUB_EVENT_NAME="pull_request_target",
                               GITHUB_REPOSITORY=REPOSITORY, GITHUB_RUN_ID="1234",
                               VALIDATION_RESULT="failure")
            result = subprocess.run(
                ["python3", str(PUBLISHER), "--dry-run"], cwd=root,
                env=environment, capture_output=True, text=True, check=False,
            )  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["head_sha"], HEAD_SHA)
        self.assertEqual(payload["conclusion"], "failure")


if __name__ == "__main__":
    unittest.main()
