"""NWS gridpoint forecast and active alerts at a point (api.weather.gov, no key, User-Agent required)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from firebinder.state import Evidence
from firebinder.tools._base import SourceStamped, utc_now_iso
from firebinder.tools._http import get_client

TOOL_NAME = "get_weather"
NWS_BASE = "https://api.weather.gov"
FORECAST_PERIODS = 6
FIRE_WEATHER_EVENTS = {"Red Flag Warning", "Fire Weather Watch"}


class ForecastPeriod(BaseModel):
    name: str
    start_time: str
    end_time: str
    temperature: float | None
    temperature_unit: str | None
    wind_speed: str | None
    wind_direction: str | None
    relative_humidity_min: float | None  # min hourly RH (%) inside the period, if available
    short_forecast: str | None


class Alert(BaseModel):
    event: str
    headline: str | None
    severity: str | None
    onset: str | None
    ends: str | None
    area: str | None


class WeatherOutlook(SourceStamped):
    point: tuple[float, float]  # (lat, lon)
    forecast_office: str | None
    periods: list[ForecastPeriod]
    alerts: list[Alert]
    red_flag_warning: bool
    fire_weather_watch: bool
    forecast_url: str
    alerts_url: str

    def to_evidence(self) -> list[Evidence]:
        return [
            self._evidence("forecast_periods", [p.model_dump() for p in self.periods], self.forecast_url),
            self._evidence("red_flag_warning", self.red_flag_warning, self.alerts_url),
            self._evidence("active_alerts", [a.event for a in self.alerts], self.alerts_url),
        ]


def _get_json(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    response = get_client().get(url, params=params, headers={"Accept": "application/geo+json"})
    response.raise_for_status()
    return response.json()


def _min_rh(hourly: list[dict[str, Any]], start: str, end: str) -> float | None:
    start_dt, end_dt = datetime.fromisoformat(start), datetime.fromisoformat(end)
    values = [
        h["relativeHumidity"]["value"]
        for h in hourly
        if (h.get("relativeHumidity") or {}).get("value") is not None
        and start_dt <= datetime.fromisoformat(h["startTime"]) < end_dt
    ]
    return min(values) if values else None


def parse_alerts(body: dict[str, Any]) -> list[Alert]:
    alerts = []
    for feature in body.get("features", []):
        p = feature.get("properties", {})
        alerts.append(
            Alert(
                event=p.get("event", ""),
                headline=p.get("headline"),
                severity=p.get("severity"),
                onset=p.get("onset"),
                ends=p.get("ends") or p.get("expires"),
                area=p.get("areaDesc"),
            )
        )
    return alerts


def fetch_weather(lat: float, lon: float) -> WeatherOutlook:
    """Next ~6 forecast periods and active alerts at (lat, lon)."""
    lat, lon = round(lat, 4), round(lon, 4)  # NWS redirects anything finer than 4 decimals
    fetched_at = utc_now_iso()

    points_url = f"{NWS_BASE}/points/{lat},{lon}"
    point = _get_json(points_url)["properties"]
    forecast_url = point["forecast"]
    hourly_url = point.get("forecastHourly")

    periods_raw = _get_json(forecast_url)["properties"]["periods"][:FORECAST_PERIODS]
    hourly = _get_json(hourly_url)["properties"]["periods"] if hourly_url else []

    periods = []
    for p in periods_raw:
        rh = (p.get("relativeHumidity") or {}).get("value")
        if rh is None:
            rh = _min_rh(hourly, p["startTime"], p["endTime"])
        periods.append(
            ForecastPeriod(
                name=p.get("name", ""),
                start_time=p["startTime"],
                end_time=p["endTime"],
                temperature=p.get("temperature"),
                temperature_unit=p.get("temperatureUnit"),
                wind_speed=p.get("windSpeed"),
                wind_direction=p.get("windDirection"),
                relative_humidity_min=rh,
                short_forecast=p.get("shortForecast"),
            )
        )

    alerts_base = f"{NWS_BASE}/alerts/active"
    alerts_params = {"point": f"{lat},{lon}"}
    alerts = parse_alerts(_get_json(alerts_base, alerts_params))
    events = {a.event for a in alerts}

    return WeatherOutlook(
        source=TOOL_NAME,
        url=points_url,
        fetched_at=fetched_at,
        point=(lat, lon),
        forecast_office=point.get("gridId"),
        periods=periods,
        alerts=alerts,
        red_flag_warning="Red Flag Warning" in events,
        fire_weather_watch="Fire Weather Watch" in events,
        forecast_url=forecast_url,
        alerts_url=f"{alerts_base}?point={lat},{lon}",
    )


class WeatherArgs(BaseModel):
    lat: float = Field(description="Latitude of the fire centroid")
    lon: float = Field(description="Longitude of the fire centroid")


@tool(TOOL_NAME, args_schema=WeatherArgs)
def get_weather(lat: float, lon: float) -> WeatherOutlook:
    """Get the NWS gridpoint forecast (next ~6 periods: temperature, wind, min RH) and active
    alerts at a point, flagging any Red Flag Warning or Fire Weather Watch."""
    return fetch_weather(lat, lon)
