"""InciWeb incident updates.

InciWeb's /feeds page lists two national feeds that work as an incident directory:
the Google Earth placemarks KML (~140 recent incidents with page links) and the
incidents RSS (the 50 most recently updated). There are no per-incident RSS feeds,
so the numbers come from the incident page itself: its "Current Situation" table
(size, containment, personnel, "Current as of") and its highlighted publications,
plus the full text of the newest publication.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

import feedparser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from firebinder.state import Evidence
from firebinder.tools._base import SourceStamped, clip, fmt, utc_now_iso
from firebinder.tools._http import get_client

TOOL_NAME = "get_inciweb_updates"
INCIWEB_BASE = "https://inciweb.wildfire.gov"
PLACEMARKS_URL = f"{INCIWEB_BASE}/feeds/maps/placemarks.kml"
INCIDENTS_RSS_URL = f"{INCIWEB_BASE}/incidents/rss.xml"
KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}

# InciWeb labels local times with the zone name ("Pacific Standard Time" even in summer).
TIME_ZONES = {
    "Eastern": "America/New_York",
    "Central": "America/Chicago",
    "Mountain": "America/Denver",
    "Arizona": "America/Phoenix",
    "Pacific": "America/Los_Angeles",
    "Alaska": "America/Anchorage",
    "Hawaii": "Pacific/Honolulu",
}

UNIT_PREFIX = re.compile(r"^([A-Z]{2})[A-Z0-9]{2,5}\s+")


class InciWebError(LookupError):
    """Incident not found in the InciWeb directory, or its page could not be parsed."""


class DirectoryEntry(BaseModel):
    title: str  # e.g. "ORMHF Grasshopper Fire"
    url: str  # incident page
    state: str | None  # two-letter code from the unit prefix


class Publication(BaseModel):
    title: str
    url: str
    published: str | None  # ISO UTC
    text: str


class InciWebUpdate(SourceStamped):
    incident_name: str  # as shown on InciWeb
    page_url: str
    size_acres: float | None
    pct_contained: float | None
    personnel: int | None
    current_as_of: str | None  # ISO, from the page's "Current as of"
    evacuation_text: str | None
    overview: str | None
    significant_events: str | None
    updates: list[Publication]  # newest first
    narrative: str | None  # full text of the newest publication

    def to_evidence(self) -> list[Evidence]:
        candidates: list[tuple[str, object]] = [
            ("acres", self.size_acres),
            ("pct_contained", self.pct_contained),
            ("personnel", self.personnel),
            ("evacuation_status", self.evacuation_text),
            ("narrative", self.narrative),
            ("inciweb_updated", self.current_as_of),
        ]
        return [self._evidence(field, value) for field, value in candidates if value is not None]

    def summary(self) -> str:
        return (
            f"InciWeb page for {self.incident_name} (current as of {fmt(self.current_as_of)}): "
            f"size {fmt(self.size_acres)} acres, containment {fmt(self.pct_contained, '%')}, "
            f"personnel {fmt(self.personnel)}. Evacuations: {clip(self.evacuation_text, 300)}. "
            f"Latest update: {clip(self.narrative)}"
        )


def html_to_text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>|</p>|</li>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    fragment = html.unescape(fragment).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in fragment.split("\n")]
    return "\n".join(line for line in lines if line)


def _absolute(url: str) -> str:
    url = url.strip()
    if url.startswith("/"):
        return INCIWEB_BASE + url
    return url.replace("http://", "https://", 1)


def _state_from_title(title: str) -> str | None:
    match = UNIT_PREFIX.match(title)
    return match.group(1) if match else None


def normalize_name(name: str) -> str:
    name = UNIT_PREFIX.sub("", name.strip())
    name = re.sub(r"\s+", " ", name).lower()
    return name.removesuffix(" fire").strip()


def parse_placemarks(kml_text: str) -> list[DirectoryEntry]:
    root = ET.fromstring(kml_text.encode())
    entries = []
    for placemark in root.iter(f"{{{KML_NS['kml']}}}Placemark"):
        title = (placemark.findtext("kml:name", default="", namespaces=KML_NS) or "").strip()
        # The description holds a raw (unescaped) <a href="..."> element, so serialize it whole.
        description_el = placemark.find("kml:description", KML_NS)
        description = ET.tostring(description_el, encoding="unicode") if description_el is not None else ""
        link = re.search(r'href="([^"]*incident-information/[^"]+)"', description)
        if title and link:
            entries.append(DirectoryEntry(title=title, url=_absolute(link.group(1)), state=_state_from_title(title)))
    return entries


def parse_incidents_rss(rss_bytes: bytes) -> list[DirectoryEntry]:
    feed = feedparser.parse(rss_bytes)
    entries = []
    for item in feed.entries:
        title = (item.get("title") or "").strip()
        link = item.get("link") or ""
        if title and "incident-information/" in link:
            entries.append(DirectoryEntry(title=title, url=_absolute(link), state=_state_from_title(title)))
    return entries


def find_incident(entries: list[DirectoryEntry], incident_name: str, state: str | None = None) -> DirectoryEntry:
    wanted = normalize_name(incident_name)
    wanted_state = state.strip().upper().removeprefix("US-") if state else None
    exact = [e for e in entries if normalize_name(e.title) == wanted]
    matches = exact or [e for e in entries if wanted in normalize_name(e.title)]
    # The same incident can appear in both feeds; dedupe by page URL.
    unique = list({e.url: e for e in matches}.values())
    if wanted_state:
        in_state = [e for e in unique if e.state == wanted_state]
        unique = in_state or unique
    if not unique:
        raise InciWebError(f"No InciWeb incident matching {incident_name!r} in the placemarks KML or incidents RSS.")
    if len(unique) > 1:
        listing = ", ".join(f"{e.title} ({e.url})" for e in unique)
        raise InciWebError(f"Several InciWeb incidents match {incident_name!r}: {listing}. Pass a state.")
    return unique[0]


def _parse_local_time(value: str, zone_label: str | None) -> str:
    """'Tue, 10/06/2026 - 15:02' in the incident's zone -> ISO with offset."""
    dt = datetime.strptime(value.strip(), "%a, %m/%d/%Y - %H:%M")
    for key, zone in TIME_ZONES.items():
        if zone_label and key.lower() in zone_label.lower():
            return dt.replace(tzinfo=ZoneInfo(zone)).isoformat()
    return dt.isoformat()


def _number(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"[\d,]+(?:\.\d+)?", text)
    return float(match.group(0).replace(",", "")) if match else None


def parse_incident_page(page_html: str) -> dict:
    """Pull the labelled table rows, overview, and highlighted publications from an incident page."""
    rows: dict[str, str] = {}
    for label, value in re.findall(r'<th scope="row">(.*?)</th>\s*<td[^>]*>(.*?)</td>', page_html, flags=re.S):
        key = html_to_text(label).rstrip(":").strip()
        rows.setdefault(key, html_to_text(value))
    if "Size" not in rows and "Percent of Perimeter Contained" not in rows:
        raise InciWebError("InciWeb incident page has no Current Situation table; the page layout may have changed.")

    title_match = re.search(r'aria-current="page">\s*<span>(.*?)</span>', page_html, flags=re.S)
    overview_match = re.search(r'class="incident-overview-copy">(.*?)</div>', page_html, flags=re.S)

    publications: list[Publication] = []
    for chunk in page_html.split('class="incident-publication-item-wrapper"')[1:]:
        link = re.search(r'<a href="(/incident-publication/[^"]+)"[^>]*>(.*?)</a>', chunk, flags=re.S)
        content = re.search(r'class="incident-publication-content">(.*?)(?:<a [^>]*views-more-link|</div>)', chunk, flags=re.S)
        when = re.search(r'<time datetime="([^"]+)"', chunk)
        if not link:
            continue
        published = when.group(1).replace("Z", "+00:00") if when else None
        publications.append(
            Publication(
                title=html_to_text(link.group(2)),
                url=_absolute(link.group(1)),
                published=published,
                text=html_to_text(content.group(1)) if content else "",
            )
        )
    publications.sort(key=lambda p: p.published or "", reverse=True)

    return {
        "title": html_to_text(title_match.group(1)) if title_match else None,
        "rows": rows,
        "overview": html_to_text(overview_match.group(1)) if overview_match else None,
        "publications": publications,
    }


def parse_publication_text(page_html: str) -> str | None:
    match = re.search(r'class="single-publication-content">(.*?)</div>', page_html, flags=re.S)
    return html_to_text(match.group(1)) if match else None


def evacuation_sentences(*texts: str | None) -> str | None:
    sentences: list[str] = []
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text or ""):
            sentence = sentence.strip()
            if re.search(r"evacuat", sentence, flags=re.I) and sentence not in sentences:
                sentences.append(sentence)
    return " ".join(sentences) or None


def _get_text(url: str) -> str:
    response = get_client().get(url)
    response.raise_for_status()
    return response.text


def fetch_inciweb_updates(incident_name: str, state: str | None = None) -> InciWebUpdate:
    """Find the incident in InciWeb's national feeds and read its incident page."""
    fetched_at = utc_now_iso()
    entries = parse_placemarks(_get_text(PLACEMARKS_URL))
    rss = get_client().get(INCIDENTS_RSS_URL)
    rss.raise_for_status()
    entries += parse_incidents_rss(rss.content)
    entry = find_incident(entries, incident_name, state)

    page = parse_incident_page(_get_text(entry.url))
    rows = page["rows"]
    publications: list[Publication] = page["publications"]

    narrative = None
    if publications:
        newest = publications[0]
        narrative = parse_publication_text(_get_text(newest.url)) or newest.text
        publications[0] = newest.model_copy(update={"text": narrative})

    current_as_of = rows.get("Current as of")
    personnel = _number(rows.get("Total Personnel"))
    significant = rows.get("Significant Events")
    return InciWebUpdate(
        source=TOOL_NAME,
        url=entry.url,
        fetched_at=fetched_at,
        incident_name=page["title"] or UNIT_PREFIX.sub("", entry.title),
        page_url=entry.url,
        size_acres=_number(rows.get("Size")),
        pct_contained=_number(rows.get("Percent of Perimeter Contained")),
        personnel=int(personnel) if personnel is not None else None,
        current_as_of=_parse_local_time(current_as_of, rows.get("Incident Time Zone")) if current_as_of else None,
        evacuation_text=evacuation_sentences(page["overview"], rows.get("Remarks"), significant, narrative),
        overview=page["overview"],
        significant_events=significant,
        updates=publications,
        narrative=narrative,
    )


class InciWebArgs(BaseModel):
    incident_name: str = Field(description="Incident name, e.g. 'Grasshopper' or 'Grasshopper Fire'")
    state: str | None = Field(default=None, description="Two-letter state code to disambiguate, e.g. 'OR'")


@tool(TOOL_NAME, args_schema=InciWebArgs, response_format="content_and_artifact")
def get_inciweb_updates(incident_name: str, state: str | None = None) -> tuple[str, InciWebUpdate]:
    """Get the public InciWeb page for an incident: reported size, percent containment, personnel,
    evacuation text, significant events, and the latest published update."""
    update = fetch_inciweb_updates(incident_name, state)
    return update.summary(), update
