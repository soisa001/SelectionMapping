"""Google Cloud Storage copy helper (gcloud/gsutil, requester-pays aware).

The controlled-tier buckets (e.g. gs://vwb-aou-datasets-controlled) are
requester-pays, so a billing project must be supplied. We use
``gcloud storage cp --billing-project=<GOOGLE_PROJECT>`` when available and
fall back to ``gsutil -u <GOOGLE_PROJECT> cp``. On the Workbench
``GOOGLE_PROJECT`` is set.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def billing_project() -> str | None:
    """The billing/user project for requester-pays access, or None."""
    return os.environ.get("GOOGLE_PROJECT") or os.environ.get("GCLOUD_PROJECT") or None


def available() -> bool:
    """True when either Workbench GCS CLI is on PATH."""
    return shutil.which("gcloud") is not None or shutil.which("gsutil") is not None


def object_metadata(url: str) -> dict | None:
    """Return remote object size and MD5, or ``None`` when unavailable."""
    proj = billing_project()
    if shutil.which("gcloud") is not None:
        cmd = [
            "gcloud", "storage", "objects", "describe", str(url),
            "--format=json(size,md5_hash)",
        ]
        if proj:
            cmd.append(f"--billing-project={proj}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            return None
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout)
            size = int(payload["size"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return {
            "size": size,
            "md5": payload.get("md5_hash") or payload.get("md5Hash"),
        }
    if shutil.which("gsutil") is not None:
        cmd = ["gsutil"]
        if proj:
            cmd += ["-u", proj]
        cmd += ["stat", str(url)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            return None
        if result.returncode != 0:
            return None
        size = re.search(r"^\s*Content-Length:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
        md5 = re.search(r"^\s*Hash \(md5\):\s*(\S+)\s*$", result.stdout, re.MULTILINE)
        if size is None:
            return None
        return {
            "size": int(size.group(1)),
            "md5": md5.group(1) if md5 is not None else None,
        }
    return None


def _local_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def gcs_same_file(local_path: str | os.PathLike, remote_url: str) -> bool:
    """Return true only when local bytes match an existing bucket object."""
    path = Path(local_path)
    try:
        metadata = object_metadata(remote_url)
        if metadata is None or path.stat().st_size != int(metadata["size"]):
            return False
        remote_md5 = metadata.get("md5")
        return bool(remote_md5) and _local_md5(path) == remote_md5
    except (KeyError, OSError, TypeError, ValueError):
        return False


def gcs_cp(src, dst) -> subprocess.CompletedProcess:
    """Copy ``src`` -> ``dst`` via gcloud storage or gsutil.

    Passes ``--billing-project`` for requester-pays buckets. Returns the
    CompletedProcess (inspect ``.returncode`` / ``.stderr``); does not raise.
    """
    proj = billing_project()
    if shutil.which("gcloud") is not None:
        cmd = ["gcloud", "storage", "cp"]
        if proj:
            cmd.append(f"--billing-project={proj}")
        cmd += [str(src), str(dst)]
    elif shutil.which("gsutil") is not None:
        cmd = ["gsutil"]
        if proj:
            cmd += ["-u", proj]
        cmd += ["cp", str(src), str(dst)]
    else:
        cmd = ["gcloud", "storage", "cp", str(src), str(dst)]
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:  # neither CLI is installed/reachable
        return subprocess.CompletedProcess(cmd, returncode=127, stdout="", stderr=str(exc))
