import pkgutil

import tiktoken_ext

# tiktoken resolves encodings from submodules of the tiktoken_ext namespace
# package with pkgutil.iter_modules, which PyInstaller's static analysis can
# not see. Enumerate the plugins here so that they are collected.
hiddenimports = [
    name
    for _, name, _ in pkgutil.iter_modules(tiktoken_ext.__path__, tiktoken_ext.__name__ + ".")
]
