"""Shared geometry helpers: local equirectangular projection and polygon area in acres.

Used by both the tools (perimeter GIS area) and reconcile (polygon area vs reported acres),
so the two code paths always agree on the number.
"""

from __future__ import annotations

from math import cos, radians

from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

M_PER_DEG_LON_AT_EQUATOR = 111_320.0
M_PER_DEG_LAT = 110_574.0
M2_PER_ACRE = 4046.8564224


def to_local_meters(geom: BaseGeometry) -> BaseGeometry:
    """Project a lon/lat geometry to local metres around its centroid latitude."""
    lat0 = geom.centroid.y
    kx = M_PER_DEG_LON_AT_EQUATOR * cos(radians(lat0))

    def project(x, y, z=None):  # type: ignore[no-untyped-def]
        return x * kx, y * M_PER_DEG_LAT

    return transform(project, geom)


def area_acres(geom: BaseGeometry) -> float:
    """Area of a lon/lat geometry in acres."""
    return to_local_meters(geom).area / M2_PER_ACRE
