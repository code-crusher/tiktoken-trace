import os
import subprocess
import sys

import pytest

import tiktoken


def test_registry_uses_namespace_package():
    # The whole reason PyInstaller support requires a hook is that
    # tiktoken_ext is a namespace package whose submodules are resolved
    # dynamically with pkgutil.iter_modules, so static analysis can not
    # discover them. If this invariant changes, our PyInstaller hook needs
    # to change too.
    prog = """
import pkgutil
import tiktoken_ext

assert not hasattr(tiktoken_ext, "__file__") or tiktoken_ext.__file__ is None
mods = [name for _, name, _ in pkgutil.iter_modules(tiktoken_ext.__path__)]
assert mods == ["openai_public"], mods
"""
    subprocess.check_call([sys.executable, "-c", prog])


def test_pyinstaller_hook_is_installed():
    # The pyinstaller40 entry point must be registered and point at a hook
    # directory that is actually shipped with the package.
    from importlib.metadata import entry_points

    import tiktoken._pyinstaller

    (hook_dir,) = tiktoken._pyinstaller.get_hook_dirs()
    assert os.path.exists(os.path.join(hook_dir, "hook-tiktoken.py"))

    eps = entry_points(group="pyinstaller40")
    ours = [ep for ep in eps if ep.module.startswith("tiktoken")]
    assert ours, f"no pyinstaller40 entry point for tiktoken found in {[ep.module for ep in eps]}"


def _build_and_run(tmp_path, prog):
    script = tmp_path / "app.py"
    script.write_text(prog)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onefile",
            "--noconfirm",
            "--distpath",
            tmp_path / "dist",
            "--workpath",
            tmp_path / "build",
            "--specpath",
            tmp_path,
            str(script),
        ],
        cwd=tmp_path,
    )
    exe = tmp_path / "dist" / "app"
    if sys.platform == "win32":
        exe = exe.with_suffix(".exe")
    return subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)


def _is_editable_install() -> bool:
    # PyInstaller can not package editable/development installs: the package
    # code lives outside of site-packages, so the module finder misses it.
    for entry in sys.path:
        if not os.path.isdir(entry):
            continue
        if any(name.startswith("__editable__") for name in os.listdir(entry)):
            if not os.path.realpath(tiktoken.__file__).startswith(os.path.realpath(entry)):
                return True
    return False


def test_pyinstaller_hook(tmp_path):
    pytest.importorskip("PyInstaller")

    if _is_editable_install():
        pytest.skip("tiktoken is installed as an editable install")

    prog = """
from tiktoken import get_encoding

get_encoding("o200k_base")
print("ok")
"""
    result = _build_and_run(tmp_path, prog)
    assert result.returncode == 0, result.stdout + result.stderr
