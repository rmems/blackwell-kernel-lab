#!/usr/bin/env python3
"""Publish the trusted GPU verdict as a required check on the candidate SHA."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from urllib import error, request


SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def require_sha(value: str) -> str:
    if not SHA.fullmatch(value):
        raise ValueError("missing or invalid candidate commit SHA")
    return value


def target_sha(event: dict, event_name: str) -> str:
    if event_name == "pull_request_target":
        return require_sha(event["pull_request"]["head"]["sha"])
    if event_name == "push":
        return require_sha(event["after"])
    if event_name == "workflow_dispatch":
        return require_sha(os.environ["GITHUB_SHA"])
    raise ValueError("unsupported event for GPU-validation publication")


def _assert_repository_identity(event: dict, repository: str) -> None:
    if not REPOSITORY.fullmatch(repository):
        raise ValueError("missing or invalid repository identity")
    if event["repository"]["full_name"] != repository:
        raise ValueError("event repository identity mismatch")


def _workflow_run_id() -> str:
    run_id = os.environ["GITHUB_RUN_ID"]
    if not run_id.isdecimal():
        raise ValueError("missing or invalid workflow run identifier")
    return run_id


def _validation_conclusion() -> tuple[str, str]:
    validation = os.environ.get("VALIDATION_RESULT", "")
    conclusion = "success" if validation == "success" else "failure"
    return validation, conclusion


def payload(event: dict, event_name: str, repository: str) -> dict:
    _assert_repository_identity(event, repository)
    run_id = _workflow_run_id()
    validation, conclusion = _validation_conclusion()
    head_sha = target_sha(event, event_name)
    return {
        "name": "GPU validation",
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": conclusion,
        "external_id": f"gpu-validation:{run_id}",
        "details_url": f"https://github.com/{repository}/actions/runs/{run_id}",
        "output": {
            "title": "Trusted GPU validation",
            "summary": (
                "The trusted GPU validation workflow completed with "
                f"`{validation or 'missing'}`. "
                "A result other than `success` is published as a failing required check."
            ),
        },
    }


def publish(check: dict, repository: str) -> None:
    token = os.environ["GITHUB_TOKEN"]
    body = json.dumps(check).encode("utf-8")
    endpoint = f"https://api.github.com/repos/{repository}/check-runs"
    submission = request.Request(
        endpoint, data=body, method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    # The endpoint is constructed from a repository value validated by payload().
    with request.urlopen(submission, timeout=30) as response:  # nosec B310
        result = json.load(response)
    if result.get("head_sha") != check["head_sha"] or result.get("name") != check["name"]:
        raise ValueError("GitHub did not create the expected GPU validation check")
    print(f"Published GPU validation check {result['id']} on {check['head_sha']}")


def main() -> int:
    try:
        mode = sys.argv[1:]
        if mode not in (["--dry-run"], ["--publish"]):
            raise ValueError("usage: publish_gpu_validation.py --dry-run|--publish")
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        event_name = os.environ["GITHUB_EVENT_NAME"]
        repository = os.environ["GITHUB_REPOSITORY"]
        check = payload(event, event_name, repository)
        if mode == ["--dry-run"]:
            print(json.dumps(check, sort_keys=True))
        else:
            publish(check, repository)
    except (KeyError, ValueError, OSError, error.URLError) as failure:
        print(f"GPU validation publication failed: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
