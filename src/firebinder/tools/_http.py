"""The one shared httpx.Client every tool uses.

15 s timeout, 3 transport-level retries, and a User-Agent header (NWS rejects
requests without one). Tests call `configure(transport=httpx.MockTransport(...))`
to serve recorded responses instead of hitting the network.
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_USER_AGENT = "FireBinder/0.1 (firebinder-dev)"
TIMEOUT_SECONDS = 15.0
RETRIES = 3

_client: httpx.Client | None = None
_transport: httpx.BaseTransport | None = None


def _build_client() -> httpx.Client:
    user_agent = os.environ.get("FIREBINDER_USER_AGENT") or DEFAULT_USER_AGENT
    transport = _transport if _transport is not None else httpx.HTTPTransport(retries=RETRIES)
    return httpx.Client(
        transport=transport,
        timeout=TIMEOUT_SECONDS,
        headers={"User-Agent": user_agent},
        follow_redirects=True,
    )


def get_client() -> httpx.Client:
    """Return the shared client, creating it on first use."""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def configure(transport: httpx.BaseTransport | None = None) -> None:
    """Swap the transport (e.g. a MockTransport in tests). None restores the real network."""
    global _client, _transport
    if _client is not None:
        _client.close()
    _client = None
    _transport = transport
