#!/usr/bin/env python3
"""Exercise the GPU gate with real Git changes and job-result inputs (CPU only)."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from gpu_ci_gate_harness import GateHarness


class GateTests(unittest.TestCase):
    def setUp(self):
        self.harness = GateHarness()
        self.addCleanup(self.harness.cleanup)

    def assert_detection(self, result, required, trusted=True, deleted=False):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.harness.read_outputs(), {
            "required": str(required).lower(), "trusted": str(trusted).lower(),
            "deleted": str(deleted).lower(),
        })

    def test_documentation_does_not_schedule_gpu(self):
        self.assert_detection(self.harness.detect_change("kernels/README.md"), False)

    def test_cpu_tool_does_not_schedule_gpu(self):
        self.assert_detection(self.harness.detect_change("tools/check_f0_efficiency.py"), False)

    def test_cuda_change_requires_gpu(self):
        self.assert_detection(self.harness.detect_change("kernels/src/operator.cu"), True)

    def test_binding_change_requires_gpu(self):
        self.assert_detection(self.harness.detect_change("bindings/torch.cpp"), True)

    def test_unknown_build_path_requires_gpu(self):
        self.assert_detection(self.harness.detect_change("pyproject.toml"), True)

    def test_unknown_files_inside_cpu_directories_require_gpu(self):
        self.harness.git("init", "-q")
        base = self.harness.commit_file("README.md", "base\n")
        for path in ("tools/new_cuda_launcher.py", "tools/setup.py",
                     "tools/custom_kernel.ptx", "tools/native_module.c",
                     "fixtures/new_operator.py", ".qlty/new_build.sh"):
            with self.subTest(path=path):
                head = self.harness.commit_file(path, "unknown executable\n")
                self.assert_detection(self.harness.detect(base, head), True)
                self.harness.output.unlink()
                base = head

    def test_fork_gate_is_data_and_cannot_change_trusted_detector(self):
        self.harness.git("init", "-q")
        base = self.harness.commit_file("README.md", "base\n")
        self.harness.commit_file("kernels/op.cu", "kernel\n")
        head = self.harness.commit_file("tools/gpu_ci_gate.py", "raise RuntimeError('untrusted')\n")
        self.assert_detection(self.harness.detect(base, head, fork=True), True, False)

    def test_gpu_workflow_requires_gpu(self):
        self.assert_detection(self.harness.detect_change(".github/workflows/ci-gpu.yml"), True)

    def test_gate_policy_change_requires_gpu(self):
        self.assert_detection(self.harness.detect_change("tools/gpu_ci_gate.py"), True)

    def test_fork_is_not_trusted(self):
        self.assert_detection(self.harness.detect_change("kernels/src/op.cu", fork=True), True, False)

    def test_fork_docs_can_pass_without_gpu(self):
        self.assert_detection(self.harness.detect_change("docs/note.md", fork=True), False, False)

    def test_renaming_cuda_to_documentation_still_requires_gpu(self):
        self.harness.git("init", "-q")
        base = self.harness.commit_file("kernels/src/op.cu", "kernel\n")
        self.harness.git("mv", "kernels/src/op.cu", "retired.md")
        self.harness.git("commit", "-qm", "rename")
        self.assert_detection(
            self.harness.detect(base, self.harness.git("rev-parse", "HEAD")), True,
        )

    def test_pr_uses_merge_base_instead_of_unrelated_base_changes(self):
        self.harness.git("init", "-q")
        common = self.harness.commit_file("README.md", "base\n")
        base = self.harness.commit_file("kernels/src/op.cu", "unrelated main change\n")
        self.harness.git("checkout", "-qb", "topic", common)
        head = self.harness.commit_file("docs/note.md", "topic\n")
        self.assert_detection(self.harness.detect(base, head), False)

    def test_push_uses_entire_pushed_range(self):
        self.harness.git("init", "-q")
        base = self.harness.commit_file("README.md", "base\n")
        self.harness.commit_file("kernels/src/op.cu", "kernel\n")
        head = self.harness.commit_file("docs/note.md", "docs\n")
        self.assert_detection(self.harness.detect(base, head, event_name="push"), True)

    def test_dispatch_requires_gpu(self):
        self.assert_detection(self.harness.detect("", "", event_name="workflow_dispatch"), True)

    def test_bootstrap_push_never_skips_gpu_after_docs_only_followup(self):
        self.harness.git("init", "-q")
        base = self.harness.commit_file("kernels/op.cu", "previous unvalidated change\n")
        head = self.harness.commit_file("docs/note.md", "followup\n")
        self.assert_detection(self.harness.detect(
            base, head, event_name="push", ref="refs/heads/codex/v020-delivery"
        ), True)

    def test_deleted_delivery_branch_does_not_schedule_or_publish_gpu_validation(self):
        result = self.harness.detect(
            "a" * 40, "0" * 40, event_name="push", ref="refs/heads/codex/v020-delivery"
        )
        self.assert_detection(result, False, deleted=True)

    def test_missing_git_commit_does_not_publish_success(self):
        result = self.harness.detect("1" * 40, "2" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.harness.output.exists())

    def test_required_gpu_result_matrix(self):
        for result in ("success", "failure", "cancelled", "skipped", "", "neutral"):
            with self.subTest(result=result):
                process = self.harness.run_gate(
                    "verify", DETECTION_RESULT="success", GPU_REQUIRED="true",
                    GPU_TRUSTED="true", GPU_RESULT=result,
                )
                self.assertEqual(process.returncode == 0, result == "success", process.stderr)

    def test_failed_detection_never_passes(self):
        for result in ("failure", "cancelled", "skipped", ""):
            with self.subTest(result=result):
                process = self.harness.run_gate(
                    "verify", DETECTION_RESULT=result, GPU_REQUIRED="false",
                    GPU_TRUSTED="true", GPU_RESULT="skipped",
                )
                self.assertNotEqual(process.returncode, 0)

    def test_missing_required_output_never_passes(self):
        process = self.harness.run_gate(
            "verify", DETECTION_RESULT="success", GPU_REQUIRED="",
            GPU_TRUSTED="true", GPU_RESULT="skipped",
        )
        self.assertNotEqual(process.returncode, 0)

    def test_untrusted_required_gpu_cannot_be_satisfied(self):
        process = self.harness.run_gate(
            "verify", DETECTION_RESULT="success", GPU_REQUIRED="true",
            GPU_TRUSTED="false", GPU_RESULT="success",
        )
        self.assertNotEqual(process.returncode, 0)

    def test_documentation_skip_passes_for_either_trust_level(self):
        for trusted in ("true", "false"):
            with self.subTest(trusted=trusted):
                process = self.harness.run_gate(
                    "verify", DETECTION_RESULT="success", GPU_REQUIRED="false",
                    GPU_TRUSTED=trusted, GPU_RESULT="skipped",
                )
                self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
