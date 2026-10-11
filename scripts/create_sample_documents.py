"""Generate reproducible fictional IT manuals from the checked-in content file."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIRECTORY = PROJECT_ROOT / "sample_docs"


def footer(canvas, document):
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#d9e0e8"))
    canvas.line(20 * mm, 20 * mm, width - 20 * mm, 20 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526172"))
    canvas.drawString(20 * mm, 14 * mm, "DOCUASSIST | FICTIONAL DEMONSTRATION MANUAL")
    canvas.drawRightString(width - 20 * mm, 14 * mm, f"PDF page {document.page}")


def main():
    definitions = json.loads((SAMPLE_DIRECTORY / "content.json").read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ManualTitle", fontName="Helvetica-Bold", fontSize=23,
                              leading=29, textColor=colors.HexColor("#173457"), spaceAfter=16))
    styles.add(ParagraphStyle("SectionTitle", fontName="Helvetica-Bold", fontSize=14,
                              leading=19, textColor=colors.HexColor("#27697c"), spaceAfter=12))
    styles.add(ParagraphStyle("ManualBody", fontName="Helvetica", fontSize=11, leading=17,
                              textColor=colors.HexColor("#263544"), alignment=TA_LEFT,
                              spaceAfter=14))
    for definition in definitions:
        target = SAMPLE_DIRECTORY / definition["filename"]
        document = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=20 * mm,
                                     rightMargin=20 * mm, topMargin=22 * mm, bottomMargin=28 * mm,
                                     title=definition["title"], author="DocuAssist project")
        story = []
        for page_index, page in enumerate(definition["pages"]):
            if page_index:
                story.append(PageBreak())
            story.append(Paragraph(escape(definition["title"]), styles["ManualTitle"]))
            story.append(Paragraph(escape(page["heading"]), styles["SectionTitle"]))
            story.append(Spacer(1, 5 * mm))
            for paragraph in page["paragraphs"]:
                story.append(Paragraph(escape(paragraph), styles["ManualBody"]))
        document.build(story, onFirstPage=footer, onLaterPages=footer)
        print(f"Created {target.name}")


if __name__ == "__main__":
    main()
