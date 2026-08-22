"""
Render a markdown document to a styled PDF using headless Edge/Chrome.

Usage:
    python src/make_pdf.py --in docs/level1_writeup.md --out docs/Level1_Writeup.pdf
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", -apple-system, Helvetica, Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; margin: 0;
}
h1 { font-size: 21pt; margin: 0 0 4pt; letter-spacing: -0.4pt; }
h2 { font-size: 14pt; margin: 20pt 0 7pt; padding-bottom: 4pt;
     border-bottom: 1.5px solid #d8d8d8; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 5pt; page-break-after: avoid; }
p { margin: 0 0 8pt; }
strong { font-weight: 600; }
code {
  font-family: Consolas, "SF Mono", Menlo, monospace; font-size: 9pt;
  background: #f4f4f6; padding: 1px 4px; border-radius: 3px;
}
pre {
  background: #f7f7f9; border: 1px solid #e4e4e8; border-radius: 5px;
  padding: 9pt 11pt; overflow-x: auto; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.5pt; line-height: 1.45; }
table {
  border-collapse: collapse; width: 100%; margin: 9pt 0 12pt;
  font-size: 9.5pt; page-break-inside: avoid;
}
th, td { border: 1px solid #dcdce0; padding: 4.5pt 8pt; text-align: left; vertical-align: top; }
th { background: #f3f3f5; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafafb; }
blockquote { margin: 0 0 8pt; padding-left: 10pt; border-left: 3px solid #d8d8d8; color: #555; }
ul, ol { margin: 0 0 9pt; padding-left: 19pt; }
li { margin-bottom: 3.5pt; }
hr { border: none; border-top: 1px solid #e0e0e4; margin: 16pt 0; }
a { color: #1a4d8f; text-decoration: none; }
h2 + table, h2 + p { margin-top: 6pt; }
"""


def find_browser():
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for name in ("msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    md_text = Path(args.inp).read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )
    html = f"<!doctype html><html><head><meta charset='utf-8'>" \
           f"<style>{CSS}</style></head><body>{html_body}</body></html>"

    browser = find_browser()
    if browser is None:
        raise SystemExit("No Edge/Chrome found to render the PDF.")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        html_file = Path(tmp) / "doc.html"
        html_file.write_text(html, encoding="utf-8")
        profile = Path(tmp) / "profile"

        cmd = [
            browser, "--headless", "--disable-gpu", "--no-sandbox",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={out_path}",
            html_file.as_uri(),
        ]
        subprocess.run(cmd, check=True, timeout=180,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not out_path.exists():
        raise SystemExit("Browser ran but produced no PDF.")
    print(f"Written {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
