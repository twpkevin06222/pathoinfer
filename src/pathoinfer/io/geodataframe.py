"""to_geodataframe(): optional interop for users who want geopandas. The only module allowed to
import geopandas; never used by core or tests. Lands in v0.1.
"""

from pathlib import Path
from typing import Any

from pathoinfer._extras import extra, require


def to_geodataframe(parquet: Path) -> Any:
    require("geopandas", extra("geopandas"))
    raise NotImplementedError("lands in v0.1")
