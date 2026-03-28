"""
Assemble paper sections into a single PDF via Markdown → HTML → PDF (WeasyPrint).
"""

import subprocess
import sys
from pathlib import Path

SECTIONS = [
    "abstract.md",
    "introduction.md",
    "related_work.md",
    "method.md",
    "results.md",
    "conclusion.md",
    "references.md",
]

TITLE = "ShiftQuant: Analyzing the Limits of Shift-Based Post-Training Quantization for LLMs"
AUTHORS = "Zorko &nbsp;·&nbsp; <a href='https://zorko.xyz'>zorko.xyz</a>"

CSS = """
@page {
    size: A4;
    margin: 2.5cm 2.8cm 2.5cm 2.8cm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #555;
    }
}

body {
    font-family: "Linux Libertine O", "Georgia", "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #111;
    text-align: justify;
    hyphens: auto;
}

h1.paper-title {
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    margin-top: 1.2cm;
    margin-bottom: 0.3cm;
    line-height: 1.3;
}

p.authors {
    text-align: center;
    font-size: 10pt;
    color: #333;
    margin-bottom: 1.2cm;
}

p.authors a {
    color: #333;
    text-decoration: none;
}

h1 {
    font-size: 12.5pt;
    font-weight: bold;
    margin-top: 1.4em;
    margin-bottom: 0.4em;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.15em;
}

h2 {
    font-size: 11pt;
    font-weight: bold;
    margin-top: 1.2em;
    margin-bottom: 0.3em;
}

h3 {
    font-size: 10.5pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 1em;
    margin-bottom: 0.3em;
}

/* Section numbering already in headings — no extra counter needed */

p {
    margin: 0.4em 0 0.6em 0;
}

pre {
    background: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 0.6em 1em;
    font-size: 8.5pt;
    line-height: 1.4;
    overflow-x: auto;
    font-family: "Courier New", monospace;
    page-break-inside: avoid;
}

code {
    font-family: "Courier New", monospace;
    font-size: 8.8pt;
    background: #f0f0f0;
    padding: 0.05em 0.25em;
    border-radius: 2px;
}

pre code {
    background: none;
    padding: 0;
    font-size: inherit;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    margin: 1em 0;
    page-break-inside: avoid;
}

th {
    background: #e8e8e8;
    font-weight: bold;
    padding: 0.35em 0.7em;
    border: 1px solid #bbb;
    text-align: center;
}

td {
    padding: 0.3em 0.7em;
    border: 1px solid #ccc;
    text-align: center;
}

tr:nth-child(even) {
    background: #fafafa;
}

strong {
    font-weight: bold;
}

em {
    font-style: italic;
}

blockquote {
    border-left: 3px solid #aaa;
    margin: 0.8em 0 0.8em 1.5em;
    padding-left: 0.8em;
    color: #444;
    font-style: italic;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 1.5em 0;
}

/* References section — hanging indent */
h1#references ~ p, h1:has(+ p > strong) ~ p {
    padding-left: 1.8em;
    text-indent: -1.8em;
    margin-bottom: 0.7em;
    font-size: 9.5pt;
}

.section-abstract {
    background: #f9f9f9;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 0.8em 1.2em;
    margin-bottom: 1.5em;
    font-size: 9.8pt;
}
"""

SECTION_LABELS = {
    "abstract.md":      "Abstract",
    "introduction.md":  "1  Introduction",
    "related_work.md":  "2  Related Work",
    "method.md":        "3  Method",
    "results.md":       "4  Experiments",
    "conclusion.md":    "5  Conclusion",
    "references.md":    "References",
}


def md_to_html(md_text: str) -> str:
    """Convert markdown to HTML using markdown_py via subprocess."""
    result = subprocess.run(
        ["markdown_py", "-x", "tables", "-x", "fenced_code"],
        input=md_text,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def build(paper_dir: Path, out_pdf: Path) -> None:
    body_parts = []

    # Title block
    body_parts.append(
        f'<h1 class="paper-title">{TITLE}</h1>\n'
        f'<p class="authors">{AUTHORS}</p>\n'
    )

    for fname in SECTIONS:
        path = paper_dir / fname
        md = path.read_text()

        # Strip the leading "# Title" line (we handle section labeling ourselves)
        lines = md.splitlines()
        if lines and lines[0].startswith("# "):
            # Replace with our canonical section heading
            label = SECTION_LABELS.get(fname, lines[0][2:])
            md = f"# {label}\n" + "\n".join(lines[1:])

        html_body = md_to_html(md)

        if fname == "abstract.md":
            body_parts.append(f'<div class="section-abstract">\n{html_body}\n</div>\n')
        else:
            body_parts.append(html_body)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{TITLE}</title>
<style>
{CSS}
</style>
</head>
<body>
{"".join(body_parts)}
</body>
</html>"""

    html_path = paper_dir / "shiftquant.html"
    html_path.write_text(html)
    print(f"  HTML written: {html_path}")

    print(f"  Rendering PDF: {out_pdf}")
    subprocess.run(
        ["weasyprint", str(html_path), str(out_pdf)],
        check=True,
    )
    print(f"  Done: {out_pdf} ({out_pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    paper_dir = Path(__file__).parent
    out_pdf = paper_dir / "shiftquant.pdf"
    build(paper_dir, out_pdf)
