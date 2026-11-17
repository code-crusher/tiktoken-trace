from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="tiktoken",
    rust_extensions=[
        RustExtension(
            "tiktoken._tiktoken",
            binding=Binding.PyO3,
            # Between our use of editable installs and wanting to use Rust for performance sensitive
            # code, it makes sense to just always use --release
            debug=False,
            features=["python"],
        )
    ],
    package_data={
        "tiktoken": ["py.typed"],
        "tiktoken._pyinstaller": ["hooks/*.py"],
    },
    packages=["tiktoken", "tiktoken._pyinstaller", "tiktoken_ext"],
    zip_safe=False,
)
