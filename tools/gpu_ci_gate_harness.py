"""Shared fixture helpers for exercising gpu_ci_gate.py in unit tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from host_cli import STDERR_DEVNULL, capture_checked, git_argv, python_script_argv, run_optional

GATE = Path(__file__).with_name("gpu_ci_gate.py")


class GateHarness:
    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.output = self.root / "outputs"
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

    def cleanup(self) -> None:
        self.directory.cleanup()

    def run_gate(self, mode: str, **values: str):
        env = dict(self.env, GITHUB_OUTPUT=str(self.output), **values)
        return run_optional(
            python_script_argv(GATE, mode),
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
        )

    def git(self, *args: str) -> str:
        output = capture_checked(
            git_argv(
                "-c", "core.hooksPath=/dev/null",
                "-c", "user.name=Gate test",
                "-c", "user.email=gate@example.invalid",
                "-c", "maintenance.auto=false",
                "-c", "gc.auto=0",
                "-c", "core.fsmonitor=false",
                *args,
            ),
            cwd=self.root,
            text=True,
            stderr=STDERR_DEVNULL,
            env=self.env,
        )
        return output.strip()

    def commit_file(self, path: str, text: str) -> str:
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        self.git("add", path)
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def write_event(
        self,
        base: str,
        head: str,
        *,
        fork: bool = False,
        event_name: str = "pull_request_target",
        ref: str = "refs/heads/main",
    ) -> Path:
        event: dict = {"repository": {"full_name": "rmems/blackwell-kernel-lab"}}
        if event_name == "pull_request_target":
            event["pull_request"] = {
                "base": {"sha": base},
                "head": {"sha": head, "repo": {
                    "full_name": "outsider/fork" if fork else "rmems/blackwell-kernel-lab"
                }},
            }
        elif event_name == "push":
            event.update(before=base, after=head, ref=ref)
        path = self.root / "event.json"
        path.write_text(json.dumps(event))
        return path

    def detect(
        self,
        base: str,
        head: str,
        *,
        fork: bool = False,
        event_name: str = "pull_request_target",
        ref: str = "refs/heads/main",
    ):
        event_path = self.write_event(
            base, head, fork=fork, event_name=event_name, ref=ref,
        )
        return self.run_gate(
            "detect",
            GITHUB_EVENT_PATH=str(event_path),
            GITHUB_EVENT_NAME=event_name,
            GITHUB_REPOSITORY="rmems/blackwell-kernel-lab",
        )

    def detect_change(self, path: str, *, fork: bool = False):
        self.git("init", "-q")
        base = self.commit_file("README.md", "base\n")
        head = self.commit_file(path, "changed\n")
        return self.detect(base, head, fork=fork)

    def read_outputs(self) -> dict[str, str]:
        return dict(line.split("=", 1) for line in self.output.read_text().splitlines())
