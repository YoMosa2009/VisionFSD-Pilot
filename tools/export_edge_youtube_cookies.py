"""Export YouTube/Google cookies from Edge into Netscape format for yt-dlp.

Uses rookiepy (works even when yt-dlp --cookies-from-browser hits DPAPI).
Writes: logs/youtube-cookies.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "youtube-cookies.txt"


def main() -> int:
    try:
        import rookiepy
    except ImportError:
        print("Installing rookiepy...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rookiepy"])
        import rookiepy

    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        cookies = rookiepy.edge()
    except Exception as exc:
        print(f"Failed to read Edge cookies: {exc}")
        print("Make sure you have used Edge to visit youtube.com while signed in.")
        return 1

    lines = [
        "# Netscape HTTP Cookie File",
        "# Auto-exported from Microsoft Edge for VisionFSD / yt-dlp",
        "",
    ]
    kept = 0
    for c in cookies:
        domain = (c.get("domain") or "").strip()
        name = c.get("name")
        value = c.get("value")
        if not domain or not name or value is None:
            continue
        dlow = domain.lower()
        if "youtube" not in dlow and "google" not in dlow:
            continue
        path = c.get("path") or "/"
        secure = "TRUE" if c.get("secure") else "FALSE"
        expires = c.get("expires") or 0
        try:
            expires = int(expires)
        except Exception:
            expires = 0
        include = "TRUE" if domain.startswith(".") else "FALSE"
        lines.append(
            f"{domain}\t{include}\t{path}\t{secure}\t{expires}\t{name}\t{value}"
        )
        kept += 1

    if kept < 5:
        print(f"Only found {kept} YouTube/Google cookies — are you signed into YouTube in Edge?")
        return 2

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {kept} cookies -> {OUT}")
    print("You can now run run_youtube_test.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
