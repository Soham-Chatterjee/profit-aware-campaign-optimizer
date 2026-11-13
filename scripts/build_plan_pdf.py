"""Render plans/implementation-plan.md to a plain PDF via headless Microsoft Edge.

Usage:  uv run --group docs python scripts/build_plan_pdf.py [input.md] [output.pdf]
Requires: the `docs` dependency group (markdown) and Microsoft Edge (or Chrome).
Relative image paths in the Markdown resolve against the input file's folder.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "plans" / "implementation-plan.md"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC.with_suffix(".pdf")

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #111; }
h1 { font-size: 19pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #bbb; padding-bottom: 2px; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 16px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: #eee; }
pre { background: #f5f5f5; border: 1px solid #ddd; padding: 8px; font-size: 8.3pt; line-height: 1.3;
      white-space: pre-wrap; page-break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt; }
ul { padding-left: 20px; } li { margin: 2px 0; }
hr { border: none; border-top: 1px solid #ccc; }
img { max-width: 100%; border: 1px solid #ddd; page-break-inside: avoid; }
"""


def find_browser() -> str:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for name in ("msedge", "chrome", "chromium", "google-chrome"):
        if p := shutil.which(name):
            return p
    sys.exit("No Edge/Chrome found for PDF rendering.")


def main() -> None:
    text = SRC.read_text(encoding="utf-8").replace("- [ ] ", "- &#9744; ")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    base = SRC.resolve().parent.as_uri() + "/"
    html = (f"<!doctype html><html><head><meta charset='utf-8'><base href='{base}'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "plan.html"
        html_path.write_text(html, encoding="utf-8")
        subprocess.run(
            [find_browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={tmp}\\profile", f"--print-to-pdf={OUT}", html_path.as_uri()],
            check=True, capture_output=True, timeout=120,
        )
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
