"""Packaging shim that ships the startup ``.pth`` autoloader.

All project metadata lives in ``pyproject.toml``; this file exists only to wire
up the one thing the declarative config cannot express: installing
``triton_cuda_backend.pth`` at the *root* of site-packages so it runs at
interpreter startup (``import _triton_cuda_autoload``) and the emit_cuda plugin
activates without any ``PYTHONPATH`` / explicit ``import``.

A ``.pth`` only works when it sits at the top level of the install dir, so we
cannot declare it as package data (that would nest it under the package). pip
installs by building a wheel and unpacking it, so overriding the legacy
``install`` command is useless — the file has to be inside the wheel's purelib.
We therefore drop it into ``build_lib`` during ``build_py`` (→ wheel root →
site-packages root) and into ``install_dir`` during ``develop`` (editable).
"""
import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

HERE = os.path.dirname(os.path.abspath(__file__))
PTH_FILE = "triton_cuda_backend.pth"


def _install_pth(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    dst = os.path.join(dest_dir, PTH_FILE)
    shutil.copyfile(os.path.join(HERE, PTH_FILE), dst)
    return dst


class BuildPyWithPth(build_py):
    """Put the .pth at the top of the wheel's purelib (covers ``pip install``)."""

    def run(self):
        super().run()
        _install_pth(self.build_lib)

    def get_outputs(self, *args, **kwargs):
        outputs = super().get_outputs(*args, **kwargs)
        outputs.append(os.path.join(self.build_lib, PTH_FILE))
        return outputs


class DevelopWithPth(develop):
    """Editable installs (``pip install -e .`` / ``setup.py develop``): copy the
    .pth straight into site-packages so the autoloader fires there too."""

    def run(self):
        super().run()
        _install_pth(self.install_dir)


setup(cmdclass={"build_py": BuildPyWithPth, "develop": DevelopWithPth})
