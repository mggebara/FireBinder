"""NASA FIRMS active-fire detections (VIIRS and MODIS) around a point, via the area CSV API.

Needs a free MAP_KEY in the FIRMS_MAP_KEY environment variable.

Sources are chosen by date from FIRMS' data-availability list (fetched once per process):
the *_NRT sources only cover the last few months, the *_SP (standard processing) archive
covers older days (VIIRS S-NPP from 2012-01-20, MODIS from 2000-11-01).
"""

from __future__ import annotations

import csv
import io
import math
import os
from datetime import date as date_type
from datetime import timedelta

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from shapely.geometry import MultiPoint

from firebinder.state import Evidence
from firebinder.tools._base import SourceStamped, fmt, utc_now_iso
from firebinder.tools._geo import area_acres
from firebinder.tools._http import get_client

TOOL_NAME = "get_hotspots"
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_AVAILABILITY_URL = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv"
# Instruments in the order they are queried; each has an _SP and/or _NRT source.
PLATFORMS = ["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21", "MODIS"]
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_AT_EQUATOR = 111.320


class FirmsError(RuntimeError):
    """FIRMS is unreachable, the key is missing/invalid, or the response is not the expected CSV."""


class SourceRange(BaseModel):
    data_id: str  # e.g. "VIIRS_SNPP_SP"
    min_date: str  # YYYY-MM-DD
    max_date: str


_availability: list[SourceRange] | None = None  # cached for the life of the process


class Detection(BaseModel):
    lat: float
    lon: float
    frp: float | None
    confidence: str
    acq_date: str  # YYYY-MM-DD (UTC)
    acq_time: str  # HHMM (UTC)
    satellite: str


class HotspotSummary(SourceStamped):
    center: tuple[float, float]  # (lat, lon)
    radius_km: float
    start_date: str
    end_date: str
    detections: list[Detection]
    count: int
    hull_acres: float
    max_frp: float | None
    latest_detection: str | None  # ISO UTC
    sources: list[str]  # FIRMS source names that responded
    urls: list[str]

    def to_evidence(self) -> list[Evidence]:
        candidates: list[tuple[str, object]] = [
            ("hotspot_count", self.count),
            ("hotspot_hull_acres", self.hull_acres),
            ("max_frp", self.max_frp),
            ("latest_detection", self.latest_detection),
        ]
        return [self._evidence(field, value) for field, value in candidates if value is not None]

    def summary(self) -> str:
        lat, lon = self.center
        return (
            f"FIRMS ({', '.join(self.sources) or 'no sources'}): {self.count} detections within "
            f"{fmt(self.radius_km)} km of ({lat}, {lon}) from {self.start_date} to {self.end_date}; "
            f"hotspot hull {fmt(self.hull_acres)} acres; max FRP {fmt(self.max_frp, ' MW')}; "
            f"latest detection {fmt(self.latest_detection)}."
        )


def bbox_around(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a square box radius_km from the point on each side."""
    dlat = radius_km / KM_PER_DEG_LAT
    dlon = radius_km / (KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat)))
    return (round(lon - dlon, 4), round(lat - dlat, 4), round(lon + dlon, 4), round(lat + dlat, 4))


def parse_firms_csv(text: str) -> list[Detection]:
    """Parse a FIRMS area CSV body. Raises FirmsError if it is not a FIRMS CSV (e.g. 'Invalid MAP_KEY')."""
    if not text.lstrip().lower().startswith("latitude"):
        raise FirmsError(f"FIRMS returned a non-CSV body: {text.strip()[:200]!r}")
    detections: list[Detection] = []
    for row in csv.DictReader(io.StringIO(text)):
        frp = row.get("frp")
        detections.append(
            Detection(
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
                frp=float(frp) if frp not in (None, "") else None,
                confidence=str(row.get("confidence", "")),
                acq_date=row["acq_date"],
                acq_time=str(row["acq_time"]).zfill(4),
                satellite=row.get("satellite", ""),
            )
        )
    return detections


def dedupe(detections: list[Detection]) -> list[Detection]:
    seen: set[tuple] = set()
    unique: list[Detection] = []
    for d in detections:
        key = (round(d.lat, 5), round(d.lon, 5), d.acq_date, d.acq_time, d.satellite)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def hull_acres(detections: list[Detection]) -> float:
    if len(detections) < 3:
        return 0.0
    hull = MultiPoint([(d.lon, d.lat) for d in detections]).convex_hull
    if hull.geom_type != "Polygon":
        return 0.0
    return area_acres(hull)


def parse_availability_csv(text: str) -> list[SourceRange]:
    if not text.lstrip().lower().startswith("data_id"):
        raise FirmsError(f"FIRMS data availability returned a non-CSV body: {text.strip()[:200]!r}")
    return [SourceRange(**row) for row in csv.DictReader(io.StringIO(text)) if row.get("data_id")]


def data_availability(map_key: str) -> list[SourceRange]:
    """Every FIRMS source with its [min_date, max_date]; fetched once per process."""
    global _availability
    if _availability is None:
        response = get_client().get(f"{FIRMS_AVAILABILITY_URL}/{map_key}/ALL")
        response.raise_for_status()
        _availability = parse_availability_csv(response.text)
    return _availability


def choose_sources(availability: list[SourceRange], start: str, end: str) -> list[str]:
    """One source per instrument for the window [start, end] (YYYY-MM-DD).

    A source qualifies if its date range overlaps the window. Per instrument, a source that
    covers the whole window beats one that covers part of it, then SP beats NRT. (NRT's
    max_date can trail today by a day, so live windows are often only partly covered.)
    """
    ranges = {r.data_id: r for r in availability}
    chosen: list[str] = []
    for platform in PLATFORMS:
        options = []
        for kind in ("SP", "NRT"):
            r = ranges.get(f"{platform}_{kind}")
            if r is None or r.min_date > end or r.max_date < start:
                continue
            full = r.min_date <= start and r.max_date >= end
            options.append((not full, kind != "SP", r.data_id))
        if options:
            chosen.append(min(options)[2])
    return chosen


def _fetch_source(map_key: str, source: str, bbox: tuple[float, float, float, float], day_range: int, start: str) -> tuple[list[Detection], str]:
    area = ",".join(str(v) for v in bbox)
    url = f"{FIRMS_AREA_URL}/{map_key}/{source}/{area}/{day_range}/{start}"
    response = get_client().get(url)
    response.raise_for_status()
    return parse_firms_csv(response.text), url


def fetch_hotspots(lat: float, lon: float, date: str, radius_km: float = 25, day_range: int = 3) -> HotspotSummary:
    """VIIRS detections within radius_km of (lat, lon) over the day_range days ending on `date` (YYYY-MM-DD)."""
    map_key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not map_key:
        raise FirmsError("FIRMS_MAP_KEY is not set. Get a free key at https://firms.modaps.eosdis.nasa.gov/api/map_key/ and put it in .env.")

    end = date_type.fromisoformat(date)
    start = end - timedelta(days=day_range - 1)
    bbox = bbox_around(lat, lon, radius_km)
    fetched_at = utc_now_iso()

    wanted = choose_sources(data_availability(map_key), start.isoformat(), end.isoformat())
    if not wanted:
        raise FirmsError(f"No FIRMS VIIRS or MODIS source covers {start.isoformat()} to {end.isoformat()}.")

    detections: list[Detection] = []
    sources: list[str] = []
    urls: list[str] = []
    first_error: Exception | None = None
    for source in wanted:
        try:
            found, url = _fetch_source(map_key, source, bbox, day_range, start.isoformat())
        except (httpx.HTTPError, FirmsError) as exc:  # one source failing is fine while another answers
            first_error = first_error or exc
            continue
        detections.extend(found)
        sources.append(source)
        urls.append(url.replace(map_key, "MAP_KEY"))  # never leak the key into evidence
    if not sources:
        raise FirmsError(f"Every FIRMS source failed ({', '.join(wanted)}): {first_error}")

    detections = dedupe(detections)
    frps = [d.frp for d in detections if d.frp is not None]
    latest = max(detections, key=lambda d: (d.acq_date, d.acq_time), default=None)
    latest_iso = f"{latest.acq_date}T{latest.acq_time[:2]}:{latest.acq_time[2:]}:00+00:00" if latest else None

    return HotspotSummary(
        source=TOOL_NAME,
        url=urls[0],
        fetched_at=fetched_at,
        center=(lat, lon),
        radius_km=radius_km,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        detections=detections,
        count=len(detections),
        hull_acres=hull_acres(detections),
        max_frp=max(frps) if frps else None,
        latest_detection=latest_iso,
        sources=sources,
        urls=urls,
    )


class HotspotsArgs(BaseModel):
    lat: float = Field(description="Latitude of the fire centroid")
    lon: float = Field(description="Longitude of the fire centroid")
    date: str = Field(description="Last day of the window, YYYY-MM-DD")
    radius_km: float = Field(default=25, description="Half-width of the search box in km")
    day_range: int = Field(default=3, ge=1, le=5, description="Number of days ending on `date`")


@tool(TOOL_NAME, args_schema=HotspotsArgs, response_format="content_and_artifact")
def get_hotspots(
    lat: float, lon: float, date: str, radius_km: float = 25, day_range: int = 3
) -> tuple[str, HotspotSummary]:
    """Get NASA FIRMS VIIRS/MODIS satellite fire detections near a point: detection count, convex-hull
    area of the detections in acres, max fire radiative power, and the latest detection time."""
    summary = fetch_hotspots(lat, lon, date, radius_km, day_range)
    return summary.summary(), summary
