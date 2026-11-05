"""to_geodataframe(): optional interop for users who want geopandas, behind the
``[geopandas]`` extra. The only module allowed to import geopandas; core and tests never
call it.
"""

from pathlib import Path
from typing import Any

import polars as pl
import shapely

from pathoinfer._extras import extra, require


def to_geodataframe(parquet: str | Path) -> Any:
    """Read an objects Parquet file into a ``geopandas.GeoDataFrame``. The geometry is in
    level-0 pixels, so the frame has no CRS."""
    gpd = require("geopandas", extra("geopandas"))
    frame = pl.read_parquet(parquet)
    geometry = shapely.from_wkb(frame["geometry"].to_numpy())
    return gpd.GeoDataFrame(frame.drop("geometry").to_pandas(), geometry=geometry, crs=None)
