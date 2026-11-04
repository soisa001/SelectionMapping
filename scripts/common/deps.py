"""Runtime dependency bootstrap for the notebooks.

A few analysis-stage packages are heavy and Workbench-only, so they are not in
the base install (see requirements.txt): ``scikit-allel`` for the Figure 3B-2G
EHH/pi VCF reads, ``dadi`` for recomputing the Fig 1 demographic fit. Call
``ensure`` from a notebook to check for them and pip-install any that are missing
-- it's a fast no-op once they're present.

    from scripts.common import deps
    deps.ensure([("allel", "scikit-allel")])       # import name, pip spec
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from shutil import which


def ensure(packages, *, quiet: bool = True) -> dict:
    """Ensure each package is importable; pip-install the missing ones.

    ``packages`` is an iterable of ``(import_name, pip_spec)`` tuples (or a bare
    string when the two match). Returns ``{pip_spec: status}`` where status is
    ``"present"``, ``"installed"`` or ``"failed"``.
    """
    results = {}
    for entry in packages:
        import_name, pip_spec = entry if isinstance(entry, tuple) else (entry, entry)
        try:
            importlib.import_module(import_name)
            results[pip_spec] = "present"
            continue
        except ImportError:
            pass
        print(f"[deps] {import_name} missing -- installing {pip_spec} ...")
        cmd = [sys.executable, "-m", "pip", "install"]
        if quiet:
            cmd.append("-q")
        cmd.append(pip_spec)
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            importlib.invalidate_caches()
            try:
                importlib.import_module(import_name)
                results[pip_spec] = "installed"
                print(f"[deps] installed {pip_spec}.")
                continue
            except ImportError:
                pass
        results[pip_spec] = "failed"
        print(f"[deps] could not install {pip_spec} (rc={rc}); "
              f"install it manually if the figure needs it.")
    return results


def have_cli(name: str) -> bool:
    """True if a command-line tool is on PATH (e.g. ``tabix``)."""
    return which(name) is not None
