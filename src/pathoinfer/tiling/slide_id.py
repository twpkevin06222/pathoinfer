"""A stable slide id that survives renames and moves (candidate to move into patchology)."""

import hashlib
from pathlib import Path

HEADER_BYTES = 1 << 20


def slide_id(path: Path, header_bytes: int = HEADER_BYTES) -> str:
    """SHA-256 (first 32 hex digits) of the file size and its first ``header_bytes``.

    Identical files get the same id wherever they live. Only the named file is read: for
    a MIRAX slide that is the ``.mrxs`` index, not its data folder.
    """
    digest = hashlib.sha256(f"{path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        digest.update(f.read(header_bytes))
    return digest.hexdigest()[:32]
