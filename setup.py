import os
import platform

import pybind11
from setuptools import Extension, setup


def get_openmp_flags():
    if platform.system() == "Darwin":
        # Apple clang does not support -fopenmp directly.
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if not conda_prefix:
            raise RuntimeError(
                "CONDA_PREFIX is not set. Activate your conda environment first."
            )
        compile_args = ["-Xpreprocessor", "-fopenmp", f"-I{conda_prefix}/include"]
        link_args = [f"-L{conda_prefix}/lib", "-lomp"]
    else:
        compile_args = ["-fopenmp"]
        link_args = ["-fopenmp"]
    return compile_args, link_args


_omp_compile, _omp_link = get_openmp_flags()

ext_modules = [
    Extension(
        "pyxdm.core.exchange_hole_cpp",
        ["pyxdm/cpp/exchange_hole.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-O3"] + _omp_compile,
        extra_link_args=_omp_link,
        language="c++",
    )
]

setup(
    name="pyxdm",
    ext_modules=ext_modules,
)
