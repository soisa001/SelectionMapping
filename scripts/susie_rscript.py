"""Run ``susieR::susie`` through Rscript without the rpy2 build dependency.

Researcher Workbench provides a runnable R installation, but its Python image
does not always expose every development linker target reported by R. Building
rpy2 can therefore fail before Figure 4 starts. This bridge exchanges NumPy
arrays with a short Rscript process through temporary little-endian float64
files, avoiding a compiled Python-to-R extension and bulky text serialization.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np


_SUSIER_CHECK = (
    'quit(status=if (requireNamespace("susieR", quietly=TRUE)) 0 else 42)'
)
_SUSIER_INSTALL = (
    'repo <- Sys.getenv("AOU_CRAN_REPO", unset="https://cloud.r-project.org"); '
    'install.packages("susieR", repos=repo)'
)
_SUSIER_READY: set[tuple[str, ...]] = set()

_RUN_SUSIE_R = r'''
args <- commandArgs(trailingOnly=TRUE)
if (!(length(args) %in% c(7L, 8L))) stop("expected n, p, L, X, y, pip, cs, [diagnostics]")

n <- as.integer(args[[1]])
p <- as.integer(args[[2]])
L <- as.integer(args[[3]])
x_path <- args[[4]]
y_path <- args[[5]]
pip_path <- args[[6]]
cs_path <- args[[7]]

read_doubles <- function(path, count) {
  con <- file(path, open="rb")
  on.exit(close(con))
  values <- readBin(con, what="double", n=count, size=8L, endian="little")
  if (length(values) != count) {
    stop(sprintf("%s contained %d doubles; expected %d", path, length(values), count))
  }
  values
}

# NumPy writes C-contiguous row-major data; byrow=TRUE reconstructs that layout.
X <- matrix(read_doubles(x_path, n * p), nrow=n, ncol=p, byrow=TRUE)
y <- read_doubles(y_path, n)
set.seed(as.integer(Sys.getenv("AOU_FIGURE_SEED", unset="20260715")))
fit_warnings <- character()
fit <- withCallingHandlers(susieR::susie(X, y, L=L), warning=function(w) {
  fit_warnings <<- c(fit_warnings, conditionMessage(w))
})

if (length(args) == 8L) {
  diagnostics <- c(
    susie_converged=as.character(isTRUE(fit$converged)),
    susie_niter=as.character(fit$niter),
    susie_version=as.character(utils::packageVersion("susieR")),
    susie_prior_variances=paste(format(fit$V, digits=17), collapse=","),
    susie_n_active_effects=as.character(sum(fit$V > 1e-9)),
    susie_elbo_final=as.character(tail(fit$elbo, 1)),
    susie_warnings=gsub("[\r\n\t]+", " ", paste(unique(fit_warnings), collapse=" | "))
  )
  write.table(data.frame(key=names(diagnostics), value=unname(diagnostics)),
              args[[8]], sep="\t", row.names=FALSE, quote=TRUE)
}

pip <- as.double(fit$pip)
if (length(pip) != p) stop("susieR returned an unexpected PIP vector length")
pip_con <- file(pip_path, open="wb")
writeBin(pip, pip_con, size=8L, endian="little")
close(pip_con)

cs <- fit$sets$cs
if (is.null(cs) || length(cs) == 0L) {
  file.create(cs_path)
} else {
  labels <- names(cs)
  if (is.null(labels)) labels <- paste0("L", seq_along(cs))
  lines <- vapply(seq_along(cs), function(i) {
    paste0(labels[[i]], "\t", paste(as.integer(cs[[i]]) - 1L, collapse=","))
  }, character(1))
  writeLines(lines, cs_path, useBytes=TRUE)
}
'''


def _rscript_command(rscript: str | os.PathLike | Sequence[str] | None = None) -> list[str]:
    """Return the executable prefix used for an Rscript subprocess."""
    if rscript is None:
        configured = os.environ.get("AOU_RSCRIPT")
        executable = configured or shutil.which("Rscript")
        if not executable:
            raise RuntimeError(
                "SuSiE recomputation requires Rscript on PATH "
                "(or set AOU_RSCRIPT to its absolute path)"
            )
        return [str(executable)]
    if isinstance(rscript, (str, os.PathLike)):
        return [str(rscript)]
    command = [str(part) for part in rscript]
    if not command:
        raise ValueError("rscript command cannot be empty")
    return command


def _run_checked(command: list[str], *, description: str) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no process output").strip()
        raise RuntimeError(f"{description} failed (rc={result.returncode}):\n{detail}")
    return result


def ensure_susie_r(
    rscript: str | os.PathLike | Sequence[str] | None = None,
) -> list[str]:
    """Ensure ``susieR`` is importable by R, installing it from CRAN if absent."""
    command = _rscript_command(rscript)
    key = tuple(command)
    if key in _SUSIER_READY:
        return command

    check = subprocess.run(
        [*command, "--vanilla", "-e", _SUSIER_CHECK],
        capture_output=True,
        text=True,
    )
    if check.returncode == 42:
        print("susieR is not available to Rscript; installing it from CRAN now.")
        _run_checked(
            [*command, "--vanilla", "-e", _SUSIER_INSTALL],
            description="susieR installation",
        )
        _run_checked(
            [*command, "--vanilla", "-e", _SUSIER_CHECK],
            description="susieR import check after installation",
        )
    elif check.returncode != 0:
        detail = (check.stderr or check.stdout or "no process output").strip()
        raise RuntimeError(
            f"Rscript could not check for susieR (rc={check.returncode}):\n{detail}"
        )

    _SUSIER_READY.add(key)
    return command


def run_susie(
    X,
    y,
    L: int = 10,
    *,
    rscript: str | os.PathLike | Sequence[str] | None = None,
    ensure_package: bool = True,
    diagnostics: dict | None = None,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Fit SuSiE in R and return ``(pip, credible_sets)`` to Python.

    ``credible_sets`` uses the same zero-based indices returned by the previous
    rpy2 wrapper, so the downstream Figure 4 code is unchanged. When supplied,
    ``diagnostics`` is populated with convergence, prior variances and warnings.
    """
    X_array = np.ascontiguousarray(X, dtype="<f8")
    y_array = np.ascontiguousarray(y, dtype="<f8").reshape(-1)
    if X_array.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {X_array.shape}")
    n, p = X_array.shape
    if len(y_array) != n:
        raise ValueError(f"y length {len(y_array)} does not match X rows {n}")
    if n == 0 or p == 0:
        raise ValueError("SuSiE requires a non-empty genotype matrix")
    if not np.isfinite(X_array).all() or not np.isfinite(y_array).all():
        raise ValueError("SuSiE inputs must be finite after residualization")
    if int(L) < 1:
        raise ValueError("L must be at least 1")

    command = ensure_susie_r(rscript) if ensure_package else _rscript_command(rscript)
    tmp_parent = os.environ.get("AOU_SUSIE_TMPDIR") or None
    with tempfile.TemporaryDirectory(prefix="aou_susie_", dir=tmp_parent) as tmp:
        tmp_dir = Path(tmp)
        x_path = tmp_dir / "X.f64"
        y_path = tmp_dir / "y.f64"
        pip_path = tmp_dir / "pip.f64"
        cs_path = tmp_dir / "credible_sets.tsv"
        script_path = tmp_dir / "run_susie.R"
        diagnostics_path = tmp_dir / "diagnostics.tsv"

        X_array.tofile(x_path)
        y_array.tofile(y_path)
        script_path.write_text(_RUN_SUSIE_R, encoding="utf-8")
        _run_checked(
            [
                *command,
                "--vanilla",
                str(script_path),
                str(n),
                str(p),
                str(int(L)),
                str(x_path),
                str(y_path),
                str(pip_path),
                str(cs_path),
                *([str(diagnostics_path)] if diagnostics is not None else []),
            ],
            description="susieR fit",
        )

        pip = np.fromfile(pip_path, dtype="<f8")
        if len(pip) != p:
            raise RuntimeError(f"susieR returned {len(pip)} PIPs for {p} variants")
        if not np.isfinite(pip).all() or ((pip < 0) | (pip > 1)).any():
            raise RuntimeError("susieR returned invalid PIP probabilities")
        if diagnostics is not None:
            with diagnostics_path.open(encoding="utf-8", newline="") as handle:
                values = {row["key"]: row["value"] for row in csv.DictReader(handle, delimiter="\t")}
            if values.get("susie_converged") not in {"TRUE", "FALSE"}:
                raise RuntimeError("susieR returned invalid convergence diagnostics")
            values["susie_converged"] = values["susie_converged"] == "TRUE"
            for key in ("susie_niter", "susie_n_active_effects"):
                values[key] = int(values[key])
            values["susie_elbo_final"] = float(values["susie_elbo_final"])
            diagnostics.update(values)
        cs_out: dict[str, list[int]] = {}
        if cs_path.exists():
            for raw_line in cs_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                name, encoded = raw_line.split("\t", 1)
                indices = [int(value) for value in encoded.split(",") if value]
                if any(index < 0 or index >= p for index in indices):
                    raise RuntimeError(f"susieR returned an invalid credible set: {raw_line}")
                cs_out[name] = indices
        return pip, cs_out
