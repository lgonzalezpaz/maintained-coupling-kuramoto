#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_manuscript_docx.py
========================
Builds journal-ready Word (.docx) documents following the exact specifications
of the `for-paper` skill (SKILL.md):
  1. Manuscript.docx (with embedded high-resolution Figures 1-3 & Tables 1-3)
  2. Manuscript_Text_Only.docx (manuscript with figure placeholders for separate upload)
  3. Supplementary Material.docx (and Supplementary_Material.docx, with Figures S1-S2 & Tables S1-S4)
  4. Response_to_Reviewers.docx (formal point-by-point response letter to Reviewers 1 & 2)

Formatting:
- A4 page (21.0 x 29.7 cm), 2.54 cm margins on all sides, single column
- Times New Roman 12 pt black body text, left-aligned, double-spaced (2.0),
  no first-line indent, space between paragraphs
- Continuous line numbers in the left margin (for Manuscript and Supplementary Material)
- Page numbers in the top-right header
- Editable centered mathematical equations with right-aligned equation numbers (1)-(17)
- High-resolution figures (300 DPI PNG) embedded at full text width (15.9 cm)
"""

from __future__ import annotations

import pathlib
import re
import shutil
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_BREAK
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from omml_equations import build_equation_omml

ROOT = pathlib.Path(__file__).resolve().parent
SERIF = "Times New Roman"
BODY_PT = 12.0


def setup_document(doc: Document, line_numbers: bool = True, body_spacing: float = 2.0):
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    for margin_attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, margin_attr, Cm(2.54))

    normal_style = doc.styles["Normal"]
    normal_style.font.name = SERIF
    normal_style.font.size = Pt(BODY_PT)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.element.rPr.rFonts.set(qn("w:eastAsia"), SERIF)

    pf = normal_style.paragraph_format
    pf.line_spacing = body_spacing
    pf.space_after = Pt(0)
    pf.first_line_indent = Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT

    if line_numbers:
        sectPr = section._sectPr
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:restart"), "continuous")
        ln.set(qn("w:distance"), "360")
        sectPr.append(ln)

    hdr = section.header.paragraphs[0]
    hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _add_page_number_field(hdr, "PAGE", size=10)


def _add_page_number_field(par, instr: str, size: int = 10):
    r = par.add_run()
    r.font.name = SERIF
    r.font.size = Pt(size)
    for tag, attr, txt in (
        ("w:fldChar", "w:fldCharType", "begin"),
        ("w:instrText", None, f" {instr} "),
        ("w:fldChar", "w:fldCharType", "end"),
    ):
        e = OxmlElement(tag)
        if attr:
            e.set(qn(attr), txt)
        else:
            e.set(qn("xml:space"), "preserve")
            e.text = txt
        r._r.append(e)


TOKEN_RE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~|`[^`]+`)")


def add_formatted_runs(par, text: str, size: float = BODY_PT, italic_all: bool = False, bold_all: bool = False):
    text = text.replace("\\*", "\u0001")
    for part in TOKEN_RE.split(text):
        if not part:
            continue
        p = part.replace("\u0001", "*")
        bold, ital, sup, sub, mono = bold_all, italic_all, False, False, False
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            p, bold = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            p, ital = part[1:-1], True
        elif part.startswith("^") and part.endswith("^") and len(part) > 2:
            p, sup = part[1:-1], True
        elif part.startswith("~") and part.endswith("~") and len(part) > 2:
            p, sub = part[1:-1], True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            p, mono = part[1:-1], True

        p = re.sub(r"\*([^*]+)\*", r"\1", p).replace("\u0001", "*")
        r = par.add_run(p)
        r.font.name = "Consolas" if mono else SERIF
        r.font.size = Pt(size - 1 if mono else size)
        r.font.color.rgb = RGBColor(0, 0, 0)
        r.bold = bold
        r.italic = ital
        r.font.superscript = sup
        r.font.subscript = sub
        if sup or sub:
            r.font.size = Pt(size * 0.78)


def add_para(
    doc: Document,
    text: str,
    size: float = BODY_PT,
    spacing: float = 2.0,
    after: float = 10.0,
    align=None,
    bold: bool = False,
    italic: bool = False,
    space_before: float = 0.0,
):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = spacing
    pf.space_after = Pt(after)
    pf.space_before = Pt(space_before)
    if align is not None:
        pf.alignment = align
    add_formatted_runs(p, text, size=size, bold_all=bold, italic_all=italic)
    return p


def add_heading(doc: Document, text: str, level: int):
    size = {1: 14.0, 2: 12.0, 3: 11.5}.get(level, 12.0)
    return add_para(doc, text, size=size, spacing=1.15, after=6, space_before=14, bold=True)


def _remove_table_borders(tbl):
    tblPr = tbl._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for b_name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{b_name}")
        b.set(qn("w:val"), "none")
        b.set(qn("w:sz"), "0")
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "auto")
        borders.append(b)
    tblPr.append(borders)


def add_equation(doc: Document, body: str, number: str):
    """Render equation in physics/mathematics journal style using native Word OMML (`<m:oMathPara>`)
    centered inside a borderless 3-column mathematical grid with right-aligned equation number."""
    omml_el = build_equation_omml(number) if number else None
    if omml_el is not None:
        tbl = doc.add_table(rows=1, cols=3)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        _remove_table_borders(tbl)

        widths = [Cm(1.6), Cm(12.7), Cm(1.6)]
        row = tbl.rows[0]
        for cell, w in zip(row.cells, widths):
            cell.width = w
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

        # Left spacer cell
        p_left = row.cells[0].paragraphs[0]
        p_left.paragraph_format.space_before = Pt(6)
        p_left.paragraph_format.space_after = Pt(6)
        p_left.paragraph_format.line_spacing = 1.15

        # Center cell: native Word OMML display equation
        p_center = row.cells[1].paragraphs[0]
        p_center.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_center.paragraph_format.space_before = Pt(6)
        p_center.paragraph_format.space_after = Pt(6)
        p_center.paragraph_format.line_spacing = 1.15
        p_center._p.append(omml_el)

        # Right cell: flush-right equation number in Times New Roman 12 pt
        p_right = row.cells[2].paragraphs[0]
        p_right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_right.paragraph_format.space_before = Pt(6)
        p_right.paragraph_format.space_after = Pt(6)
        p_right.paragraph_format.line_spacing = 1.15
        add_formatted_runs(p_right, number, size=BODY_PT)
        return tbl

    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(8)
    pf.space_after = Pt(8)
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.tab_stops.add_tab_stop(Cm(7.95), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Cm(15.9), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t")
    add_formatted_runs(p, body, size=BODY_PT)
    p.add_run("\t")
    add_formatted_runs(p, number, size=BODY_PT)
    return p


def add_table(doc: Document, rows: list[list[str]]):
    if not rows:
        return None
    hdr, body = rows[0], rows[1:]
    t = doc.add_table(rows=1, cols=len(hdr))
    t.style = "Table Grid"
    t.autofit = True
    for c, txt in zip(t.rows[0].cells, hdr):
        c.paragraphs[0].paragraph_format.line_spacing = 1.05
        c.paragraphs[0].paragraph_format.space_after = Pt(2)
        add_formatted_runs(c.paragraphs[0], txt, size=9.0, bold_all=True)
    for row in body:
        cells = t.add_row().cells
        for c, txt in zip(cells, row):
            c.paragraphs[0].paragraph_format.line_spacing = 1.05
            c.paragraphs[0].paragraph_format.space_after = Pt(2)
            add_formatted_runs(c.paragraphs[0], txt, size=9.0)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    return t


def embed_figure_image(doc: Document, png_path: pathlib.Path, width_cm: float = 15.9):
    if png_path.exists():
        pf = doc.add_paragraph()
        pf.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.paragraph_format.space_before = Pt(8)
        pf.paragraph_format.space_after = Pt(6)
        pf.add_run().add_picture(str(png_path), width=Cm(width_cm))


def convert_markdown_to_docx(
    md_path: pathlib.Path,
    out_path: pathlib.Path,
    embed_figures: bool = True,
    line_numbers: bool = True,
    body_spacing: float = 2.0,
):
    md = md_path.read_text(encoding="utf-8")
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    lines = md.split("\n")

    doc = Document()
    setup_document(doc, line_numbers=line_numbers, body_spacing=body_spacing)

    fig_map = {
        "Figure 1.": ROOT / "Figure1.png",
        "Figure 2.": ROOT / "Figure2.png",
        "Figure 3.": ROOT / "Figure3.png",
        "Supplementary Figure S1.": ROOT / "FigureS1.png",
        "Supplementary Figure S2.": ROOT / "FigureS2.png",
    }

    i = 0
    state = None
    tbl_buf = []

    while i < len(lines):
        ln = lines[i].rstrip()

        # Table parsing
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not re.fullmatch(r"[-: ]+", "".join(cells)):
                tbl_buf.append(cells)
            i += 1
            if not (i < len(lines) and lines[i].rstrip().startswith("|")):
                add_table(doc, tbl_buf)
                tbl_buf = []
            continue

        if not ln.strip() or ln.strip() == "---":
            i += 1
            continue

        if ln.startswith("# "):
            add_para(doc, ln[2:], size=16.0, spacing=1.15, after=12, bold=True)
            state = "title"
            i += 1
            continue

        if ln.startswith("## "):
            h = ln[3:]
            add_heading(doc, h, 1)
            if h.lower().startswith("abstract"):
                state = "abstract"
            elif h.lower().startswith("references"):
                state = "refs"
            else:
                state = "body"
            i += 1
            continue

        if ln.startswith("### "):
            add_heading(doc, ln[4:].strip(), 2)
            i += 1
            continue

        # Blockquote or Equation
        if ln.startswith("> "):
            body = ln[2:]
            m = re.search(r"\s{2,}(\([S]?\d+\))\s*$", body)
            if m:
                num = m.group(1)
                body = body[: m.start()]
                add_equation(doc, body.strip().rstrip(","), num)
            else:
                # Reviewer quote block in Response_to_Reviewers.md
                p = add_para(doc, body.strip(), size=10.5, spacing=1.25, after=8, italic=True)
                p.paragraph_format.left_indent = Cm(0.8)
            i += 1
            continue

        if state == "title":
            if "Alvarado" in ln or "González" in ln:
                add_para(doc, ln, size=12.0, spacing=1.15, after=6)
            else:
                add_para(doc, ln, size=9.5, spacing=1.15, after=4)
            i += 1
            continue

        if state == "abstract":
            if ln.startswith("**Keywords:**"):
                add_para(doc, ln, size=9.5, spacing=1.15, after=12)
            else:
                add_para(doc, ln, size=10.5, spacing=1.15, after=10)
            i += 1
            continue

        if state == "refs":
            m = re.match(r"^(\d+)\.\s+(.*)$", ln)
            if m:
                p = add_para(doc, f"{m.group(1)}.\t{m.group(2)}", size=9.5, spacing=1.15, after=4)
                p.paragraph_format.left_indent = Cm(0.9)
                p.paragraph_format.first_line_indent = Cm(-0.9)
                i += 1
                continue

        # Handle figure captions and figure embedding
        if (
            ln.startswith("**[")
            or ln.startswith("**Figure")
            or ln.startswith("**Table")
            or ln.startswith("**Supplementary")
        ):
            if embed_figures and (ln.startswith("**Figure") or ln.startswith("**Supplementary Figure")):
                for prefix, png_file in fig_map.items():
                    if ln.startswith(f"**{prefix}"):
                        embed_figure_image(doc, png_file, width_cm=15.9)
                        break
            add_para(doc, ln, size=10.5, spacing=1.15, after=10)
            i += 1
            continue

        add_para(doc, ln, spacing=body_spacing, after=10)
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    print(f"Saved: {out_path} ({out_path.stat().st_size:,} bytes)")


def sync_legacy_figure_filenames():
    """Ensure both `Figure1.png` and `Figure 1.png` exist for GitHub/Zenodo compatibility."""
    pairs = [
        ("Figure1.png", "Figure 1.png"),
        ("Figure2.png", "Figure 2.png"),
        ("Figure3.png", "Figure 3.png"),
        ("FigureS1.png", "Figure S1.png"),
        ("FigureS2.png", "Figure S2.png"),
    ]
    for src_name, dst_name in pairs:
        src = ROOT / src_name
        dst = ROOT / dst_name
        if src.exists():
            shutil.copy2(src, dst)


if __name__ == "__main__":
    sync_legacy_figure_filenames()

    # 1. Main Manuscript with embedded high-res figures
    convert_markdown_to_docx(
        md_path=ROOT / "Manuscript.md",
        out_path=ROOT / "Manuscript.docx",
        embed_figures=True,
        line_numbers=True,
        body_spacing=2.0,
    )
    shutil.copy2(ROOT / "Manuscript.docx", ROOT / "Manuscript_Complete_with_Figures.docx")

    # 2. Text-only Manuscript (for journals requiring separate figure uploads)
    convert_markdown_to_docx(
        md_path=ROOT / "Manuscript.md",
        out_path=ROOT / "Manuscript_Text_Only.docx",
        embed_figures=False,
        line_numbers=True,
        body_spacing=2.0,
    )

    # 3. Supplementary Material with embedded Figures S1 & S2 and Tables S1-S4
    convert_markdown_to_docx(
        md_path=ROOT / "Supplementary_Material.md",
        out_path=ROOT / "Supplementary Material.docx",
        embed_figures=True,
        line_numbers=True,
        body_spacing=1.5,
    )
    shutil.copy2(ROOT / "Supplementary Material.docx", ROOT / "Supplementary_Material.docx")

    # 4. Point-by-Point Response to Reviewers
    convert_markdown_to_docx(
        md_path=ROOT / "Response_to_Reviewers.md",
        out_path=ROOT / "Response_to_Reviewers.docx",
        embed_figures=False,
        line_numbers=False,
        body_spacing=1.35,
    )
