"""Local, lazy CPU OCR with packaged models and bounded PDF rasterization."""
import logging
import math
from contextlib import closing
from importlib.util import find_spec

from backend.config import Settings

logger = logging.getLogger(__name__)


class OCRUnavailableError(RuntimeError):
    pass


def render_scale(width: float, height: float, dpi: int, max_pixels: int) -> float:
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("Invalid PDF page dimensions")
    scale = min(dpi / 72, math.sqrt(max_pixels / (width * height)))
    # PDFium rounds each dimension up; account for that before allocating a bitmap.
    while math.ceil(width * scale) * math.ceil(height * scale) > max_pixels:
        scale *= 0.99
    return scale


class LocalOCR:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine = None
        self._initialization_failed = False

    def status(self) -> dict:
        dependencies = all(find_spec(name) is not None for name in (
            "rapidocr_onnxruntime", "onnxruntime", "pypdfium2", "numpy"
        ))
        state = (
            "disabled" if not self.settings.ocr_enabled else
            "failed" if self._initialization_failed else
            "unavailable" if not dependencies else
            "ready" if self._engine is not None else "not_loaded"
        )
        return {"enabled": self.settings.ocr_enabled, "state": state,
                "engine": "RapidOCR / ONNX Runtime", "dependencies_available": dependencies}

    def _get_engine(self):
        if self._initialization_failed:
            raise OCRUnavailableError("OCR initialization failed; restart the backend after repair.")
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._engine = RapidOCR(
                    intra_op_num_threads=2, inter_op_num_threads=1,
                    det_limit_side_len=1280, det_limit_type="max",
                    max_side_len=2500, print_verbose=False,
                    # The line-angle classifier flipped an upright English prerequisite
                    # in our real scan fixture. This version expects upright scans.
                    use_cls=False,
                )
            except Exception as error:
                self._initialization_failed = True
                logger.exception("Local OCR engine initialization failed")
                raise OCRUnavailableError(
                    "Local OCR is unavailable. Install requirements.txt and restart the backend."
                ) from error
        return self._engine

    def read_page(self, content: bytes, page_index: int) -> str:
        engine = self._get_engine()
        try:
            import numpy as np
            import pypdfium2 as pdfium
        except (ImportError, OSError) as error:
            raise OCRUnavailableError("PDF rendering dependencies are unavailable.") from error

        with pdfium.PdfDocument(content) as pdf:
            with closing(pdf[page_index]) as page:
                width, height = page.get_size()
                scale = render_scale(width, height, self.settings.ocr_dpi,
                                     self.settings.ocr_max_pixels)
                with closing(page.render(scale=scale)) as bitmap:
                    # RapidOCR's ndarray input uses OpenCV BGR channel order.
                    with bitmap.to_pil().convert("RGB") as image:
                        pixels = np.asarray(image)[:, :, ::-1].copy()
        result, _ = engine(pixels)
        # The engine returns detected lines in reading order, with box/text/score entries.
        return "\n".join(line[1] for line in (result or []) if line[1].strip())
