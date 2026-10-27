"""NIFC WFIGS incident record: the Locations (points) layer plus the Interagency Perimeters layer.

Both are public ArcGIS FeatureServers, no key. Dates come back as epoch milliseconds.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import shapely
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from firebinder.state import Evidence
from firebinder.tools._base import SourceStamped, fmt, utc_now_iso
from firebinder.tools._http import get_client

TOOL_NAME = "get_incident_record"

ARCGIS_HOST = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
LOCATIONS_QUERY_URL = f"{ARCGIS_HOST}/WFIGS_Incident_Locations_Current/FeatureServer/0/query"
PERIMETERS_QUERY_URL = f"{ARCGIS_HOST}/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query"

LOCATION_FIELDS = [
    "IncidentName",
    "IrwinID",
    "IncidentSize",
    "PercentContained",
    "FireDiscoveryDateTime",
    "POOState",
    "POOCounty",
    "TotalIncidentPersonnel",
    "InitialLatitude",
    "InitialLongitude",
    "ModifiedOnDateTime_dt",
    "ICS209ReportDateTime",
    "IncidentTypeCategory",
    "CpxName",
    "UniqueFireIdentifier",
]

PERIMETER_FIELDS = [
    "poly_IRWINID",
    "poly_IncidentName",
    "poly_GISAcres",
    "poly_Acres_AutoCalc",
    "poly_DateCurrent",
    "poly_PolygonDateTime",
    "poly_MapMethod",
    "attr_IncidentSize",
    "attr_PercentContained",
]


class WfigsError(RuntimeError):
    """The WFIGS service returned an error or no matching record."""


def epoch_ms_to_iso(value: int | float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat(timespec="seconds")


def arcgis_query(query_url: str, params: dict[str, str]) -> tuple[dict[str, Any], str]:
    """Run an ArcGIS REST query. Returns (json body, full request URL)."""
    response = get_client().get(query_url, params={**params, "f": "json"})
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise WfigsError(f"ArcGIS error from {query_url}: {body['error']}")
    return body, str(response.url)


def esri_rings_to_geometry(rings: list[list[list[float]]]) -> Polygon | MultiPolygon:
    """Convert ESRI polygon rings (lon, lat) to a shapely geometry.

    ESRI convention: clockwise rings are exteriors, counter-clockwise rings are holes.
    Each hole is attached to the exterior that contains it.
    """
    exteriors: list[list[list[float]]] = []
    holes: list[list[list[float]]] = []
    for ring in rings:
        if len(ring) < 4:
            continue
        if LinearRing(ring).is_ccw:
            holes.append(ring)
        else:
            exteriors.append(ring)

    hole_lists: list[list[list[list[float]]]] = [[] for _ in exteriors]
    exterior_polys = [Polygon(ring) for ring in exteriors]
    for hole in holes:
        point = Polygon(hole).representative_point()
        for i, poly in enumerate(exterior_polys):
            if poly.contains(point):
                hole_lists[i].append(hole)
                break

    polygons = [orient(Polygon(ext, hole_lists[i]), sign=1.0) for i, ext in enumerate(exteriors)]
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def geometry_to_geojson(geom: BaseGeometry) -> dict[str, Any]:
    return json.loads(shapely.to_geojson(geom))


class IncidentRecord(SourceStamped):
    irwin_id: str
    incident_name: str
    state: str | None = None
    county: str | None = None
    irwin_daily_acres: float | None = None
    gis_acres: float | None = None
    pct_contained: float | None = None
    personnel: int | None = None
    discovery_date: str | None = None
    record_modified: str | None = None
    ics209_report_time: str | None = None
    centroid: tuple[float, float] | None = None  # (lat, lon)
    perimeter_date: str | None = None
    map_method: str | None = None
    perimeter_geometry: dict[str, Any] | None = None  # GeoJSON, lon/lat
    locations_url: str
    perimeters_url: str

    def to_evidence(self) -> list[Evidence]:
        candidates: list[tuple[str, object, str]] = [
            ("acres", self.irwin_daily_acres, self.locations_url),
            ("gis_acres", self.gis_acres, self.perimeters_url),
            ("pct_contained", self.pct_contained, self.locations_url),
            ("personnel", self.personnel, self.locations_url),
            ("discovery_date", self.discovery_date, self.locations_url),
            ("perimeter_geometry", self.perimeter_geometry, self.perimeters_url),
            ("perimeter_date", self.perimeter_date, self.perimeters_url),
        ]
        return [self._evidence(field, value, url) for field, value, url in candidates if value is not None]

    def summary(self) -> str:
        perimeter = f"perimeter polygon dated {fmt(self.perimeter_date)}" if self.perimeter_geometry else "no perimeter polygon"
        return (
            f"WFIGS record for {self.incident_name} ({fmt(self.state)}, IRWIN {self.irwin_id}): "
            f"IRWIN daily acres {fmt(self.irwin_daily_acres)}, GIS acres {fmt(self.gis_acres)}, "
            f"containment {fmt(self.pct_contained, '%')}, personnel {fmt(self.personnel)}, "
            f"discovered {fmt(self.discovery_date)}; {perimeter}. "
            "Structures and evacuations are not in this record."
        )


def fetch_incident_record(irwin_id: str) -> IncidentRecord:
    """Fetch the WFIGS point record and (if one exists) the newest perimeter for an IRWIN ID."""
    irwin_id = irwin_id.strip()
    if not irwin_id.startswith("{"):
        irwin_id = "{" + irwin_id + "}"
    safe_id = irwin_id.replace("'", "''")
    fetched_at = utc_now_iso()

    loc_body, locations_url = arcgis_query(
        LOCATIONS_QUERY_URL,
        {
            "where": f"IrwinID = '{safe_id}'",
            "outFields": ",".join(LOCATION_FIELDS),
            "returnGeometry": "true",
            "outSR": "4326",
        },
    )
    loc_features = loc_body.get("features", [])
    if not loc_features:
        raise WfigsError(f"No WFIGS incident location found for IRWIN ID {irwin_id}")
    # Several point records for one IRWIN ID would be odd; take the most recently modified.
    loc_feature = max(loc_features, key=lambda f: f["attributes"].get("ModifiedOnDateTime_dt") or 0)
    loc = loc_feature["attributes"]

    per_body, perimeters_url = arcgis_query(
        PERIMETERS_QUERY_URL,
        {
            "where": f"poly_IRWINID = '{safe_id}'",
            "outFields": ",".join(PERIMETER_FIELDS),
            "returnGeometry": "true",
            "outSR": "4326",
        },
    )
    per_features = [f for f in per_body.get("features", []) if f.get("geometry", {}).get("rings")]

    gis_acres = None
    perimeter_date = None
    map_method = None
    perimeter_geojson = None
    centroid: tuple[float, float] | None = None

    if per_features:
        newest = max(per_features, key=lambda f: f["attributes"].get("poly_DateCurrent") or 0)
        attrs = newest["attributes"]
        gis_acres = attrs.get("poly_GISAcres")
        if gis_acres is None:
            gis_acres = attrs.get("poly_Acres_AutoCalc")
        perimeter_date = epoch_ms_to_iso(attrs.get("poly_DateCurrent"))
        map_method = attrs.get("poly_MapMethod")
        geom = esri_rings_to_geometry(newest["geometry"]["rings"])
        perimeter_geojson = geometry_to_geojson(geom)
        centroid = (geom.centroid.y, geom.centroid.x)
    else:
        point = loc_feature.get("geometry") or {}
        if "x" in point and "y" in point:
            centroid = (point["y"], point["x"])
        elif loc.get("InitialLatitude") is not None and loc.get("InitialLongitude") is not None:
            centroid = (loc["InitialLatitude"], loc["InitialLongitude"])

    personnel = loc.get("TotalIncidentPersonnel")
    return IncidentRecord(
        source=TOOL_NAME,
        url=locations_url,
        fetched_at=fetched_at,
        irwin_id=loc.get("IrwinID") or irwin_id,
        incident_name=loc.get("IncidentName") or "",
        state=loc.get("POOState"),
        county=loc.get("POOCounty"),
        irwin_daily_acres=loc.get("IncidentSize"),
        gis_acres=gis_acres,
        pct_contained=loc.get("PercentContained"),
        personnel=int(personnel) if personnel is not None else None,
        discovery_date=epoch_ms_to_iso(loc.get("FireDiscoveryDateTime")),
        record_modified=epoch_ms_to_iso(loc.get("ModifiedOnDateTime_dt")),
        ics209_report_time=epoch_ms_to_iso(loc.get("ICS209ReportDateTime")),
        centroid=centroid,
        perimeter_date=perimeter_date,
        map_method=map_method,
        perimeter_geometry=perimeter_geojson,
        locations_url=locations_url,
        perimeters_url=perimeters_url,
    )


class IncidentRecordArgs(BaseModel):
    irwin_id: str = Field(description="IRWIN ID of the incident, e.g. '{6AA4C355-AD18-472F-AAAB-78BAA9DE31B1}'")


@tool(TOOL_NAME, args_schema=IncidentRecordArgs, response_format="content_and_artifact")
def get_incident_record(irwin_id: str) -> tuple[str, IncidentRecord]:
    """Get the NIFC WFIGS incident record for an IRWIN ID: IRWIN daily acres, GIS (perimeter)
    acres, percent contained, personnel, discovery date, and the current perimeter polygon."""
    record = fetch_incident_record(irwin_id)
    return record.summary(), record
