import pathlib
import subprocess
import sys

import tiktoken


def test_encoding_for_model():
    enc = tiktoken.encoding_for_model("gpt2")
    assert enc.name == "gpt2"
    enc = tiktoken.encoding_for_model("text-davinci-003")
    assert enc.name == "p50k_base"
    enc = tiktoken.encoding_for_model("text-davinci-edit-001")
    assert enc.name == "p50k_edit"
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    assert enc.name == "cl100k_base"
    enc = tiktoken.encoding_for_model("gpt-4")
    assert enc.name == "cl100k_base"
    enc = tiktoken.encoding_for_model("gpt-4o")
    assert enc.name == "o200k_base"
    enc = tiktoken.encoding_for_model("gpt-oss-120b")
    assert enc.name == "o200k_harmony"


def test_optional_blobfile_dependency():
    prog = """
import tiktoken
import sys
assert "blobfile" not in sys.modules
"""
    subprocess.check_call([sys.executable, "-c", prog])


def test_linux_wheel_is_manylinux2014_compatible():
    # Wheels built by cibuildwheel must install on manylinux2014 systems
    # (glibc 2.17), e.g. Amazon Linux 2. This runs against the installed
    # wheel during cibuildwheel's test phase. See issue #456.
    if not sys.platform.startswith("linux"):
        return

    # look for the dist-info next to the imported tiktoken package, so we
    # inspect the wheel actually under test rather than some other tiktoken
    # installation on the system
    package_parent = pathlib.Path(tiktoken.__file__).parent.parent
    dist_info_dirs = list(package_parent.glob("tiktoken-*.dist-info"))

    # an editable or source-tree install has no wheel under test
    if not dist_info_dirs:
        return

    wheel_text = (dist_info_dirs[0] / "WHEEL").read_text()
    tags = [
        line.split(": ", 1)[1].strip()
        for line in wheel_text.splitlines()
        if line.startswith("Tag:")
    ]
    manylinux_tags = [tag for tag in tags if "manylinux" in tag]
    if not manylinux_tags or all("musllinux" in tag for tag in manylinux_tags):
        return

    assert any(
        "manylinux2014" in tag or "manylinux_2_17" in tag for tag in manylinux_tags
    ), f"wheel is not manylinux2014-compatible: {tags}"
