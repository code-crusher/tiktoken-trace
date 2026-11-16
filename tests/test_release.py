"""Tests for the PyPI publishing workflow.

Releases must be published to PyPI with Trusted Publishing (OIDC) from CI
instead of with manually-managed, long-lived API tokens, which are a supply
chain risk. See https://docs.pypi.org/trusted-publishers/
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS_DIR = Path(__file__).parent.parent / ".github" / "workflows"
PUBLISH_WORKFLOW_PATH = WORKFLOWS_DIR / "publish.yml"
BUILD_WORKFLOW_PATH = WORKFLOWS_DIR / "build_wheels.yml"


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def publish_workflow():
    return _load(PUBLISH_WORKFLOW_PATH)


@pytest.fixture(scope="module")
def build_workflow():
    return _load(BUILD_WORKFLOW_PATH)


@pytest.fixture(scope="module")
def publish_steps(publish_workflow):
    return publish_workflow["jobs"]["publish"]["steps"]


def test_publish_workflow_exists():
    assert PUBLISH_WORKFLOW_PATH.exists(), (
        "releases must be uploaded to PyPI from CI with trusted publishing"
    )


def test_publish_workflow_is_manual_dispatch_only(publish_workflow):
    # publishing must never be triggered automatically by pushes or PRs
    triggers = publish_workflow[True]
    assert triggers == ["workflow_dispatch"]


def test_publish_job_permissions_are_least_privilege(publish_workflow):
    workflow = publish_workflow
    job = workflow["jobs"]["publish"]

    # no permissions are granted outside of the id-token needed for OIDC
    assert workflow.get("permissions") == {}
    assert job["permissions"] == {"id-token": "write"}


def test_publish_job_requests_oidc_token(publish_workflow):
    assert publish_workflow["jobs"]["publish"]["permissions"]["id-token"] == "write"


def test_publish_job_uses_no_secrets(publish_steps):
    for step in publish_steps:
        for key in ("env", "with"):
            for value in step.get(key, {}).values():
                assert "secrets." not in str(value), "the publish job must not use any secrets"


def test_publish_job_uses_no_static_credentials(publish_steps):
    env_keys = {key for step in publish_steps for key in step.get("env", {})}
    assert "PYPI_API_TOKEN" not in env_keys, "PyPI API tokens must not be used"
    assert "TWINE_PASSWORD" not in env_keys, "PyPI API tokens must not be used"

    uses = [step.get("uses", "") for step in publish_steps]
    assert not any("password" in u.lower() or "api-token" in u.lower() for u in uses)


def test_publish_job_uses_official_pypa_action(publish_steps):
    publish_actions = [
        step["uses"] for step in publish_steps if "gh-action-pypi-publish" in step.get("uses", "")
    ]
    assert publish_actions, "publishing must use pypa/gh-action-pypi-publish"
    # no API token input, so the action can only authenticate with OIDC
    for step in publish_steps:
        assert not step.get("with", {}).get("password"), "no PyPI token may be passed"
        assert not step.get("with", {}).get("api-token"), "no PyPI token may be passed"


def test_publish_job_downloads_cibw_artifacts(publish_workflow):
    step = next(
        step
        for step in publish_workflow["jobs"]["publish"]["steps"]
        if step.get("uses", "").startswith("actions/download-artifact")
    )
    # the wheels and sdist built by the build workflow
    assert step["with"] == {"pattern": "cibw-wheels", "path": "dist", "merge-multiple": True}


def test_build_workflow_joins_artifacts(build_workflow):
    join_job = next(
        job for job in build_workflow["jobs"].values() if job.get("name") == "Join artifacts"
    )
    merge_step = next(step for step in join_job["steps"] if "merge" in step.get("uses", ""))
    assert merge_step["with"]["name"] == "cibw-wheels"
    assert merge_step["with"]["pattern"] == "cibw-wheels-*"
    assert merge_step["with"]["delete-merged"] is True


SHA_PATTERN = re.compile(r"[0-9a-f]{40}$")


@pytest.mark.parametrize(
    "workflow_path",
    [PUBLISH_WORKFLOW_PATH, BUILD_WORKFLOW_PATH],
    ids=["publish", "build_wheels"],
)
def test_workflow_actions_are_pinned_by_sha(workflow_path):
    workflow = _load(workflow_path)
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if not uses:  # plain run steps
                continue
            ref = uses.split("@")[-1]
            assert SHA_PATTERN.match(ref), f"action {uses} must be pinned by full commit SHA"


def test_publish_workflow_actions_are_trusted(publish_steps):
    for step in publish_steps:
        uses = step.get("uses", "")
        assert uses.startswith(("actions/", "pypa/")), (
            f"only actions/* or pypa/* actions are allowed, got {uses}"
        )
