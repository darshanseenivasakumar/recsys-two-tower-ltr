"""Download, verify and extract a MovieLens archive.

Production habits shown here:
- Idempotent: re-running is a no-op if the data is already extracted.
- Atomic: download to a `.part` file and rename only when complete, so a crash
  never leaves a truncated archive that looks valid.
- Verified: compare against the MD5 checksum GroupLens publishes.
- Safe extraction: refuse zip entries that would escape the target folder ("zip slip").
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import certifi

from recsys.data.datasets import DatasetSpec

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1 << 20  # 1 MiB


def _ssl_context() -> ssl.SSLContext:
    # Use Mozilla's CA bundle (certifi) instead of the OS store: files.grouplens.org
    # chains to a newer root CA that some Windows/Python installs don't ship yet.
    # We keep certificate verification ON; never "fix" this with verify=False.
    return ssl.create_default_context(cafile=certifi.where())


def md5_of_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    with (
        urllib.request.urlopen(url, context=_ssl_context(), timeout=60) as resp,
        open(tmp, "wb") as out,
    ):
        shutil.copyfileobj(resp, out, length=_CHUNK_BYTES)
    tmp.replace(dest)  # atomic rename on the same filesystem


def safe_extract(archive: Path, target_dir: Path) -> None:
    """Extract a zip, rejecting any entry whose resolved path escapes `target_dir`."""
    target = target_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            dest = (target / member).resolve()
            if not dest.is_relative_to(target):
                raise ValueError(f"Unsafe path in archive: {member!r}")
        zf.extractall(target)


def download_dataset(spec: DatasetSpec, raw_dir: Path, force: bool = False) -> Path:
    """Ensure `spec`'s files exist under raw_dir/<name>/; return the extracted folder."""
    dataset_dir = raw_dir / spec.name
    extracted = dataset_dir / spec.extracted_dir
    if extracted.is_dir() and not force:
        logger.info("%s already present at %s; skipping download", spec.name, extracted)
        return extracted

    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive = dataset_dir / Path(urlparse(spec.url).path).name
    if force or not archive.exists():
        _stream_download(spec.url, archive)

    actual = md5_of_file(archive)
    if actual != spec.md5:
        archive.unlink()  # don't leave a corrupt file that a re-run would trust
        raise ValueError(f"MD5 mismatch for {archive.name}: expected {spec.md5}, got {actual}")
    logger.info("Checksum OK (%s)", actual)

    safe_extract(archive, dataset_dir)
    return extracted
