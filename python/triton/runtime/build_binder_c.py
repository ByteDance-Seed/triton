"""Standalone builder for the ``_binder_c`` CPython extension.

Normally you don't need to run this by hand: ``Triton-distributed/python/setup.py``
declares ``triton.runtime._binder_c`` in ``ext_modules``, so ``pip install`` /
``pip wheel`` build and ship the ``.so`` as part of the wheel just like
``libtriton.so``.

This script is kept around as a developer convenience for fast iteration
on ``_binder_c.c`` without re-running the full Triton build::

    python3 build_binder_c.py

It compiles ``_binder_c.c`` and drops the resulting ``.so`` next to this
file.  The extension is optional at runtime — if it is missing, ``jit.py``
falls back to the legacy ``exec``'d ``dynamic_func`` and behaviour is
unchanged.
"""

from __future__ import annotations

import os
import sys
import shutil
import sysconfig
import tempfile
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


HERE = Path(__file__).resolve().parent


def main() -> int:
    src = HERE / "_binder_c.c"
    if not src.exists():
        print(f"missing {src}", file=sys.stderr)
        return 1

    ext = Extension(
        "_binder_c",
        sources=[str(src)],
        extra_compile_args=["-O2", "-fvisibility=hidden"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        old_argv = sys.argv
        sys.argv = [
            "build_binder_c.py",
            "build_ext",
            "--build-lib", tmp,
            "--build-temp", os.path.join(tmp, "tmp"),
            "--force",
        ]
        try:
            setup(
                name="_binder_c",
                ext_modules=[ext],
                cmdclass={"build_ext": build_ext},
                script_args=sys.argv[1:],
            )
        finally:
            sys.argv = old_argv

        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        produced = list(Path(tmp).glob(f"_binder_c*{suffix}"))
        if not produced:
            produced = list(Path(tmp).glob("_binder_c*"))
        if not produced:
            print(f"build failed: no _binder_c.* found in {tmp}", file=sys.stderr)
            return 2
        dst = HERE / produced[0].name
        shutil.copy2(produced[0], dst)
        print(f"built {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
