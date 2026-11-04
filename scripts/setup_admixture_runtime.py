"""Create and register a clean, self-contained admixture reference Jupyter runtime."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements-admixture.txt"
RUNTIME_CONTRACT = "numpy2-phetk-isolated-v1"
DEFAULT_KERNEL_NAME = "allofus-figure4"
DEFAULT_DISPLAY_NAME = "Python (All of Us admixture reference)"
FALLBACK_SKLEARN_VERSION = "1.9.0"


def _runtime_root() -> Path:
    override = os.environ.get("AOU_FIG4_RUNTIME_DIR")
    if override:
        return Path(override).expanduser().resolve()
    python_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return (Path.home() / ".venvs" /
            f"allofus-figure4-{RUNTIME_CONTRACT}-{python_tag}")


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _source_sklearn_version() -> str:
    try:
        return version("scikit-learn")
    except PackageNotFoundError:
        return FALLBACK_SKLEARN_VERSION


def _run(command: list[str], *, cwd: Path | None = None,
         env: dict[str, str] | None = None,
         capture: bool = False) -> subprocess.CompletedProcess:
    displayed = [
        "<python code>" if index and command[index - 1] == "-c" else part
        for index, part in enumerate(command)
    ]
    print("[admixture_reference-runtime]", " ".join(displayed), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def _repo_environment(phetk_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["AOU_FIG4_PHETK_ENV"] = str(phetk_dir)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _validate_python_runtime(python: Path, env: dict[str, str]) -> None:
    code = """
import importlib.metadata as metadata
import numpy
packages = [
    'numpy', 'pandas', 'matplotlib', 'scipy', 'polars', 'statsmodels',
    'scikit-learn', 'cyvcf2', 'ipykernel',
]
assert numpy.__version__ == '2.0.2', numpy.__version__
for package in packages:
    print(f'{package}=={metadata.version(package)}')
from scripts import admixture_finemap, admixture_lai, admixture_phewas, admixture_plrs
print('admixture reference Python imports: OK')
"""
    _run([str(python), "-c", code], cwd=REPO_ROOT, env=env)
    _run([str(python), "-m", "pip", "check"], env=env)


def _build_phetk_runtime(python: Path, phetk_dir: Path,
                         env: dict[str, str]) -> None:
    code = """
from scripts.admixture_phewas import _ensure_phetk_runtime
print(_ensure_phetk_runtime())
"""
    _run([str(python), "-c", code], cwd=REPO_ROOT, env=env)
    phetk_python = _venv_python(phetk_dir)
    check = """
import importlib.metadata as metadata
import numpy
assert metadata.version('phetk') == '0.3.3'
assert numpy.__version__ == '1.26.4', numpy.__version__
from phetk.phewas import PheWAS
print(f'phetk=={metadata.version("phetk")}, numpy=={numpy.__version__}: OK')
"""
    _run([str(phetk_python), "-c", check], env=env)


def _register_kernel(python: Path, phetk_dir: Path, *, kernel_name: str,
                     display_name: str, env: dict[str, str]) -> Path:
    install_code = (
        "from ipykernel.kernelspec import install; "
        f"print(install(user=True, kernel_name={kernel_name!r}, "
        f"display_name={display_name!r}))"
    )
    result = _run(
        [str(python), "-c", install_code],
        env=env,
        capture=True,
    )
    output_lines = [line.strip() for line in result.stdout.splitlines()
                    if line.strip()]
    if not output_lines:
        raise RuntimeError(f"Registration of kernel {kernel_name!r} returned no path")
    resource_dir = Path(output_lines[-1])
    spec_path = resource_dir / "kernel.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_env = spec.setdefault("env", {})
    spec_env.update({
        "AOU_FIG4_PHETK_ENV": str(phetk_dir),
        "PYTHONNOUSERSITE": "1",
    })
    temporary = spec_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    temporary.replace(spec_path)
    return resource_dir


def build_runtime(*, runtime_root: Path, kernel_name: str,
                  display_name: str, sklearn_version: str,
                  install_r: bool = True) -> dict[str, Path]:
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("admixture reference requires Python 3.10 through 3.12")
    runtime_root = runtime_root.expanduser().resolve()
    kernel_venv = runtime_root / "kernel"
    phetk_venv = runtime_root / "phetk"
    runtime_root.mkdir(parents=True, exist_ok=True)

    kernel_python = _venv_python(kernel_venv)
    if not kernel_python.exists():
        _run([sys.executable, "-m", "venv", str(kernel_venv)])
    _run([
        str(kernel_python), "-m", "pip", "install",
        "--disable-pip-version-check", "--upgrade", "pip",
    ])
    _run([
        str(kernel_python), "-m", "pip", "install",
        "--disable-pip-version-check", "--upgrade",
        "--upgrade-strategy", "eager", "-r", str(REQUIREMENTS),
        f"scikit-learn=={sklearn_version}",
    ])

    env = _repo_environment(phetk_venv)
    _validate_python_runtime(kernel_python, env)
    _build_phetk_runtime(kernel_python, phetk_venv, env)
    if install_r:
        _run([
            str(kernel_python), "-c",
            "from scripts.susie_rscript import ensure_susie_r; "
            "print('Rscript:', *ensure_susie_r())",
        ], cwd=REPO_ROOT, env=env)
    kernelspec = _register_kernel(
        kernel_python, phetk_venv,
        kernel_name=kernel_name,
        display_name=display_name,
        env=env,
    )
    manifest = runtime_root / "runtime.json"
    manifest.write_text(json.dumps({
        "contract": RUNTIME_CONTRACT,
        "kernel_python": str(kernel_python),
        "phetk_python": str(_venv_python(phetk_venv)),
        "scikit_learn_version": sklearn_version,
        "kernelspec": str(kernelspec),
    }, indent=2), encoding="utf-8")
    print("\nadmixture reference runtime is ready.")
    print(f"Select this Jupyter kernel: {display_name}")
    print(f"Runtime manifest: {manifest}")
    return {
        "runtime_root": runtime_root,
        "kernel_python": kernel_python,
        "phetk_python": _venv_python(phetk_venv),
        "kernelspec": kernelspec,
        "manifest": manifest,
    }


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=_runtime_root())
    parser.add_argument("--kernel-name", default=DEFAULT_KERNEL_NAME)
    parser.add_argument("--display-name", default=DEFAULT_DISPLAY_NAME)
    parser.add_argument(
        "--scikit-learn-version", default=_source_sklearn_version(),
        help="Version used to read existing Stage 2 model pickles.",
    )
    parser.add_argument(
        "--skip-r", action="store_true",
        help="Do not validate/install the external R package susieR.",
    )
    args = parser.parse_args(argv)
    return build_runtime(
        runtime_root=args.runtime_dir,
        kernel_name=args.kernel_name,
        display_name=args.display_name,
        sklearn_version=args.scikit_learn_version,
        install_r=not args.skip_r,
    )


if __name__ == "__main__":
    main()
