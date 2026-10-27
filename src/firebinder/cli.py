"""FireBinder command-line interface."""

from __future__ import annotations

import typer

app = typer.Typer(help="FireBinder: draft and audit wildfire ICS-209 summaries from public data.")


@app.callback()
def main() -> None:
    """FireBinder CLI."""


if __name__ == "__main__":
    app()
