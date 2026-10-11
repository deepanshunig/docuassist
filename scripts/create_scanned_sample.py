"""Create an image-only PDF from our fictional VPN manual for real OCR checks."""
from pathlib import Path
from contextlib import closing

import pypdfium2 as pdfium
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "sample_docs" / "fictional-vpn-guide.pdf"
TARGET = PROJECT_ROOT / "sample_docs" / "fictional-scanned-vpn-guide.pdf"


def main():
    canvas = Canvas(str(TARGET), invariant=1, pageCompression=1)
    canvas.setTitle("Fictional VPN guide - image-only OCR sample")
    canvas.setAuthor("DocuAssist project")
    with pdfium.PdfDocument(SOURCE) as pdf:
        for page_index in range(len(pdf)):
            with closing(pdf[page_index]) as page:
                width, height = page.get_size()
                canvas.setPageSize((width, height))
                with closing(page.render(scale=180 / 72)) as bitmap:
                    with bitmap.to_pil().convert("RGB") as image:
                        canvas.drawImage(ImageReader(image), 0, 0, width=width, height=height)
                canvas.showPage()
    canvas.save()
    assert all(not page.extract_text().strip() for page in PdfReader(TARGET).pages)
    print(f"Created image-only PDF: {TARGET}")


if __name__ == "__main__":
    main()
