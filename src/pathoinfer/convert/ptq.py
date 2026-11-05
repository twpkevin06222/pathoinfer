"""INT8 / FP8 post-training quantisation, calibrated on tiles read through patchology. Lands in
v0.2.
"""

from pathlib import Path

from pathoinfer._extras import extra, require


def quantise(folder: Path, out: Path) -> Path:
    require("modelopt", extra("convert"))
    raise NotImplementedError("lands in v0.2")
