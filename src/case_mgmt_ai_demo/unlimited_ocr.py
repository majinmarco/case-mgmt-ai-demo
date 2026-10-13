"""Unlimited-OCR (baidu/Unlimited-OCR) as an OCR engine for text drawn as outlines.

The model reads an image into text but does not say where each word is, so an outlined run
cannot be picked out of a page reading by position. Instead the run is read together with
the printed text on either side of it, as one image, and the reading of the run is what
lies between those two texts. Printed text that the model does not read back as the PDF
has it leaves the run unread: the neighbours are the check on the reading.

The model runs on a GPU, in process, with its own code from Hugging Face (pinned to a
revision). It needs the `unlimited-ocr` extra: `uv sync --extra unlimited-ocr`.
"""

import contextlib
import io
import re
import tempfile
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pymupdf

from case_mgmt_ai_demo.extraction import Outline

MODEL = "baidu/Unlimited-OCR"
REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"

_PROMPT = "<image>document parsing."
_DPI = 300
_MARGIN = 40  # white pixels around the crop: text at the very edge of an image is misread
_ABOVE_AND_BELOW = 3.0  # points of the page kept over and under the run
_MAX_TOKENS = 1024  # one line of text, with the image
_BLOCK_MARKER = re.compile(r"<\|det\|>.*?<\|/det\|>")
_NOT_TEXT = re.compile(r"\\[\[(]|<\|")  # a formula in LaTeX, or a marker left in the reading
_RUN = "\0"
_MISREAD_NEIGHBOUR = 3  # characters of printed text next to the run that may differ
_BESIDE = 12  # characters of printed text on each side of the run that locate it
_READ_BACK = 0.7  # how much of them the reading must have, in order

type ImageReader = Callable[[Path], str]


class UnlimitedOcrUnavailable(Exception):
    """The model cannot run here; the message says what is missing."""


class UnlimitedOcr:
    name = "unlimited-ocr"

    def __init__(self, read_image: ImageReader | None = None):
        """`read_image` stands in for the model in tests; left out, the model is loaded."""
        self._read_image = read_image or _load_model()

    def read(self, page: pymupdf.Page, outlines: Sequence[Outline]) -> list[str]:
        with tempfile.TemporaryDirectory(prefix="unlimited-ocr-") as directory:
            image = Path(directory) / "line.png"
            readings: list[str] = []
            for outline in outlines:
                _crop(page, outline).save(image)
                readings.append(
                    reading_between(self._read_image(image), outline.before, outline.after)
                )
            return readings


def reading_between(line: str, before: str | None, after: str | None) -> str:
    """What `line`, the model's reading of a run with its neighbours, has between the
    printed text `before` and `after` the run; empty when it cannot be told."""
    line = _typographic_quotes(_spaced(_BLOCK_MARKER.sub(" ", line)))
    if _NOT_TEXT.search(line):
        return ""
    left, right = _spaced(before or ""), _spaced(after or "")
    printed = left + _RUN + right
    start, end = 0, len(line)
    unmatched_left, unmatched_right = left, right
    read_back = [False] * len(printed)
    for block in SequenceMatcher(None, printed, line, autojunk=False).get_matching_blocks():
        read_back[block.a : block.a + block.size] = [True] * block.size
        if block.size and block.a < len(left):
            start = block.b + block.size
            unmatched_left = left[block.a + block.size :]
        elif block.size and unmatched_right == right:
            end = block.b
            unmatched_right = right[: block.a - len(left) - 1]
    beside = read_back[max(0, len(left) - _BESIDE) : len(left) + 1 + _BESIDE]
    unmatched_left, unmatched_right = unmatched_left.strip(), unmatched_right.strip()
    if (
        sum(beside) < _READ_BACK * (len(beside) - 1)
        or max(len(unmatched_left), len(unmatched_right)) > _MISREAD_NEIGHBOUR
    ):
        return ""
    # Printed text read a little differently beside the run is taken to be as long as printed.
    reading = line[start:end].strip()
    return reading[len(unmatched_left) : len(reading) - len(unmatched_right)].strip()


def _spaced(text: str) -> str:
    return " ".join(text.split())


def _typographic_quotes(text: str) -> str:
    """The Reglamento prints “ and ”; the model often reads either as a straight quote."""
    return re.sub(r'(?<![^\s(\[¿¡])"', "“", text).replace('"', "”")


def _crop(page: pymupdf.Page, outline: Outline) -> pymupdf.Pixmap:
    clip = pymupdf.Rect(
        outline.context_x0 - 1,
        outline.y0 - _ABOVE_AND_BELOW,
        outline.context_x1 + 1,
        outline.y1 + _ABOVE_AND_BELOW,
    )
    line = page.get_pixmap(dpi=_DPI, clip=clip, colorspace=pymupdf.csRGB)
    line.set_origin(_MARGIN, _MARGIN)
    framed = pymupdf.Pixmap(
        pymupdf.csRGB, (0, 0, line.width + 2 * _MARGIN, line.height + 2 * _MARGIN)
    )
    framed.clear_with(255)
    framed.copy(line, line.irect)
    return framed


def _load_model() -> ImageReader:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise UnlimitedOcrUnavailable(
            "faltan sus dependencias: ejecute `uv sync --extra unlimited-ocr`"
        ) from error
    if not torch.cuda.is_available():
        raise UnlimitedOcrUnavailable("PyTorch no encuentra una GPU, y el modelo la requiere")
    # MIOpen (ROCm) has no bfloat16 convolution; PyTorch's own is used instead.
    torch.backends.cudnn.enabled = False
    tokenizer: Any = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    model: Any = AutoModel.from_pretrained(
        MODEL,
        revision=REVISION,
        trust_remote_code=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
    )
    model = model.eval().cuda()

    def read_image(image: Path) -> str:
        # The model's code prints as it goes and wants a directory to write to.
        with contextlib.redirect_stdout(io.StringIO()):
            return model.infer(
                tokenizer,
                prompt=_PROMPT,
                image_file=str(image),
                output_path=str(image.parent),
                base_size=1024,
                image_size=1024,
                crop_mode=False,
                max_length=_MAX_TOKENS,
                no_repeat_ngram_size=35,
                ngram_window=128,
                eval_mode=True,
            )

    return read_image
