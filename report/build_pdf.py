"""Renders report.md (with embedded figures) to report.pdf.

Requires the report-building dependencies (markdown, xhtml2pdf):
    pip install -r requirements-report.txt
"""

from pathlib import Path

import markdown
from xhtml2pdf import pisa

REPORT_DIR = Path(__file__).resolve().parent
MD_PATH = REPORT_DIR / "report.md"
PDF_PATH = REPORT_DIR / "report.pdf"

md_text = MD_PATH.read_text(encoding="utf-8")
body_html = markdown.markdown(md_text, extensions=["tables", "sane_lists"])

CSS = """
@page {
    size: letter;
    margin: 2.2cm 2.2cm 2.2cm 2.2cm;
    @frame footer_frame {
        -pdf-frame-content: footer_content;
        bottom: 1cm; margin-left: 2.2cm; margin-right: 2.2cm; height: 1cm;
    }
}
body { font-family: Helvetica, sans-serif; font-size: 10.3pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 16pt; margin-top: 0; margin-bottom: 6pt; }
h2 { font-size: 13pt; margin-top: 16pt; margin-bottom: 5pt; border-bottom: 0.75pt solid #999999; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin-top: 10pt; margin-bottom: 4pt; }
p { margin-top: 0pt; margin-bottom: 7pt; text-align: justify; }
strong { font-weight: bold; }
em { font-style: italic; }
code { font-family: Courier, monospace; font-size: 9.3pt; background-color: #f2f2f2; }
pre { font-family: Courier, monospace; font-size: 9pt; background-color: #f2f2f2; padding: 6pt; }
hr { color: #bbbbbb; margin-top: 10pt; margin-bottom: 10pt; }
table { width: 100%; border-collapse: collapse; margin-top: 6pt; margin-bottom: 10pt; }
th, td { border: 0.5pt solid #888888; padding: 4pt 6pt; font-size: 9.6pt; text-align: left; }
th { background-color: #eeeeee; }
img { max-width: 100%; margin-top: 8pt; margin-bottom: 2pt; }
"""

html_doc = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div id="footer_content" style="text-align:center; font-size:8pt; color:#888888;">
    <pdf:pagenumber/>
</div>
{body_html}
</body>
</html>
"""

with open(PDF_PATH, "wb") as f:
    result = pisa.CreatePDF(html_doc, dest=f, path=str(REPORT_DIR) + "/")

if result.err:
    raise RuntimeError(f"{result.err} error(s) while rendering PDF")
print("wrote", PDF_PATH)
