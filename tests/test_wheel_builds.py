"""Tests for the riscv64 wheel build in .github/workflows/build_wheels.yml.

Skipped when running from an installed wheel or an sdist, neither of which
ships the workflow file.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "build_wheels.yml"

pytestmark = pytest.mark.skipif(
    not WORKFLOW.exists(), reason="wheel build workflow is not available"
)


def _jobs() -> dict[str, str]:
    """Return the text of each job in the workflow, keyed by job name."""
    jobs: dict[str, list[str]] = {}
    current: list[str] | None = None
    in_jobs = False
    for line in WORKFLOW.read_text().splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            in_jobs = line == "jobs:"
            current = None
            continue
        if not in_jobs:
            continue
        if match := re.fullmatch(r"  ([A-Za-z0-9_-]+):", line):
            current = jobs.setdefault(match.group(1), [])
        elif current is not None:
            current.append(line)
    return {name: "\n".join(lines) for name, lines in jobs.items()}


def _artifact_name(job: str) -> str:
    match = re.search(r"^\s+name: (cibw-wheels-.+)$", _jobs()[job], re.MULTILINE)
    assert match is not None, f"{job} does not upload a cibw-wheels artifact"
    return match.group(1).strip()


def test_riscv64_job_builds_manylinux_wheels():
    jobs = _jobs()
    assert "build_wheels_riscv64" in jobs

    job = jobs["build_wheels_riscv64"]
    # cibuildwheel cross-builds for riscv64, emulated via QEMU
    assert re.search(r"^\s+CIBW_ARCHS: riscv64$", job, re.MULTILINE)
    assert re.search(r"^\s+platforms: riscv64$", job, re.MULTILINE)
    # Rust is installed inside the manylinux container, like the other jobs
    assert re.search(r"^\s+CIBW_ENVIRONMENT_LINUX: .*\$HOME/\.cargo/bin", job, re.MULTILINE)
    # Wheels are uploaded from the cibuildwheel output directory
    assert re.search(r"^\s+path: \./wheelhouse/\*\.whl$", job, re.MULTILINE)


def test_riscv64_wheels_are_joined_with_the_rest():
    job = _jobs()["join_artifacts"]
    needs = re.search(r"^\s+needs: \[(.*)\]$", job, re.MULTILINE)
    assert needs is not None
    assert "build_wheels_riscv64" in [name.strip() for name in needs.group(1).split(",")]

    # The riscv64 upload must be picked up by the artifact merge
    artifact = _artifact_name("build_wheels_riscv64").replace("${{ strategy.job-index }}", "0")
    pattern = re.search(r"^\s+pattern: (\S+)$", job, re.MULTILINE)
    assert pattern is not None
    assert fnmatch.fnmatchcase(artifact, pattern.group(1))
