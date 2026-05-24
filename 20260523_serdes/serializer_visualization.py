#!/usr/bin/env python3
"""Launch the interactive pure SerDes serializer visualization."""

from pathlib import Path
import webbrowser


def main() -> None:
    html = Path(__file__).with_name("serializer_visualization.html").resolve()
    if not html.exists():
        raise SystemExit(f"Missing {html.name}. Keep it next to this launcher.")
    webbrowser.open(html.as_uri())
    print(f"Opened {html}")


if __name__ == "__main__":
    main()
