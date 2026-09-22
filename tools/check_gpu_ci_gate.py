#!/usr/bin/env python3
"""Exercise the GPU gate with real Git changes and job-result inputs (CPU only)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

GATE = Path(__file__).with_name("gpu_ci_gate.py")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "outputs"
        # Fixture repositories must not inherit signing, hooks, or asynchronous
        # trace consumers that write notes after a test starts removing its repo.
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

    def run_gate(self, mode, **values):
        env = dict(self.env, GITHUB_OUTPUT=str(self.output), **values)
        return subprocess.run(
            ["python3", str(GATE), mode], cwd=self.root, env=env,
            capture_output=True, text=True, check=False,
        )

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "user.name=Gate test",
             "-c", "user.email=gate@example.invalid", "-c", "maintenance.auto=false",
             "-c", "gc.auto=0", "-c", "core.fsmonitor=false", *args],
            cwd=self.root, text=True, stderr=subprocess.DEVNULL, env=self.env,
        ).strip()

    def commit_file(self, path, text):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        self.git("add", path)
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def detect_change(self, path, *, fork=False):
        self.git("init", "-q")
        base = self.commit_file("README.md", "base\n")
        head = self.commit_file(path, "changed\n")
        return self.detect(base, head, fork=fork)

    def detect(self, base, head, *, fork=False, event_name="pull_request"):
        event = {"repository": {"full_name": "rmems/blackwell-kernel-lab"}}
        if event_name == "pull_request":
            event["pull_request"] = {
                "base": {"sha": base},
                "head": {"sha": head, "repo": {
                    "full_name": "outsider/fork" if fork else "rmems/blackwell-kernel-lab"
                }},
            }
        elif event_name == "push":
            event.update(before=base, after=head)
        path = self.root / "event.json"
        path.write_text(json.dumps(event))
        return self.run_gate(
            "detect", GITHUB_EVENT_PATH=str(path), GITHUB_EVENT_NAME=event_name,
            GITHUB_REPOSITORY="rmems/blackwell-kernel-lab",
        )

    def assert_detection(self, result, required, trusted=True):
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = dict(line.split("=", 1) for line in self.output.read_text().splitlines())
        self.assertEqual(outputs, {
            "required": str(required).lower(), "trusted": str(trusted).lower(),
        })

    def test_documentation_does_not_schedule_gpu(self):
        self.assert_detection(self.detect_change("kernels/README.md"), False)

    def test_cpu_tool_does_not_schedule_gpu(self):
        self.assert_detection(self.detect_change("tools/check_f0_efficiency.py"), False)

    def test_cuda_change_requires_gpu(self):
        self.assert_detection(self.detect_change("kernels/src/operator.cu"), True)

    def test_binding_change_requires_gpu(self):
        self.assert_detection(self.detect_change("bindings/torch.cpp"), True)

    def test_unknown_build_path_requires_gpu(self):
        self.assert_detection(self.detect_change("pyproject.toml"), True)

    def test_gpu_workflow_requires_gpu(self):
        self.assert_detection(self.detect_change(".github/workflows/ci-gpu.yml"), True)

    def test_gate_policy_change_requires_gpu(self):
        self.assert_detection(self.detect_change("tools/gpu_ci_gate.py"), True)

    def test_fork_is_not_trusted(self):
        self.assert_detection(self.detect_change("kernels/src/op.cu", fork=True), True, False)

    def test_fork_docs_can_pass_without_gpu(self):
        self.assert_detection(self.detect_change("docs/note.md", fork=True), False, False)

    def test_renaming_cuda_to_documentation_still_requires_gpu(self):
        self.git("init", "-q")
        base = self.commit_file("kernels/src/op.cu", "kernel\n")
        self.git("mv", "kernels/src/op.cu", "retired.md")
        self.git("commit", "-qm", "rename")
        self.assert_detection(self.detect(base, self.git("rev-parse", "HEAD")), True)

    def test_pr_uses_merge_base_instead_of_unrelated_base_changes(self):
        self.git("init", "-q")
        common = self.commit_file("README.md", "base\n")
        base = self.commit_file("kernels/src/op.cu", "unrelated main change\n")
        self.git("checkout", "-qb", "topic", common)
        head = self.commit_file("docs/note.md", "topic\n")
        self.assert_detection(self.detect(base, head), False)

    def test_push_uses_entire_pushed_range(self):
        self.git("init", "-q")
        base = self.commit_file("README.md", "base\n")
        self.commit_file("kernels/src/op.cu", "kernel\n")
        head = self.commit_file("docs/note.md", "docs\n")
        self.assert_detection(self.detect(base, head, event_name="push"), True)

    def test_dispatch_requires_gpu(self):
        self.assert_detection(self.detect("", "", event_name="workflow_dispatch"), True)

    def test_missing_git_commit_does_not_publish_success(self):
        result = self.detect("1" * 40, "2" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_required_gpu_result_matrix(self):
        for result in ("success", "failure", "cancelled", "skipped", "", "neutral"):
            with self.subTest(result=result):
                process = self.run_gate(
                    "verify", DETECTION_RESULT="success", GPU_REQUIRED="true",
                    GPU_TRUSTED="true", GPU_RESULT=result,
                )
                self.assertEqual(process.returncode == 0, result == "success", process.stderr)

    def test_failed_detection_never_passes(self):
        for result in ("failure", "cancelled", "skipped", ""):
            with self.subTest(result=result):
                process = self.run_gate(
                    "verify", DETECTION_RESULT=result, GPU_REQUIRED="false",
                    GPU_TRUSTED="true", GPU_RESULT="skipped",
                )
                self.assertNotEqual(process.returncode, 0)

    def test_missing_required_output_never_passes(self):
        process = self.run_gate(
            "verify", DETECTION_RESULT="success", GPU_REQUIRED="",
            GPU_TRUSTED="true", GPU_RESULT="skipped",
        )
        self.assertNotEqual(process.returncode, 0)

    def test_untrusted_required_gpu_cannot_be_satisfied(self):
        process = self.run_gate(
            "verify", DETECTION_RESULT="success", GPU_REQUIRED="true",
            GPU_TRUSTED="false", GPU_RESULT="success",
        )
        self.assertNotEqual(process.returncode, 0)

    def test_documentation_skip_passes_for_either_trust_level(self):
        for trusted in ("true", "false"):
            with self.subTest(trusted=trusted):
                process = self.run_gate(
                    "verify", DETECTION_RESULT="success", GPU_REQUIRED="false",
                    GPU_TRUSTED=trusted, GPU_RESULT="skipped",
                )
                self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
