"""The Unlimited-OCR engine, with the model's readings given by hand (the model is not run)."""

import pytest

from case_mgmt_ai_demo.extraction import GAP, extract_lines
from case_mgmt_ai_demo.unlimited_ocr import UnlimitedOcr, reading_between
from conftest import RC_PDF

FOOTER_TOP = 738


@pytest.mark.parametrize(
    ("line", "before", "after", "reading"),
    [
        # Readings the model gave for lines of RC-2023-E.
        (
            "Contiene las capas de como construido (“AsBuilt”) de la",
            "Contiene las capas de como construido (",
            ") de la ",
            "“AsBuilt”",
        ),
        (
            "8. Cada uno de estos deben estar hechos en capas (“layers”) individuales y en polígonos",
            "8. Cada uno de ",
            "individuales y en polígonos ",
            "estos deben estar hechos en capas (“layers”)",
        ),
        (
            "Pluvial” vigente, adoptada por la JP.",
            None,
            None,
            "Pluvial” vigente, adoptada por la JP.",
        ),
        (
            "a. La Ley 161-2009, según enmendada, conocida como la “Ley para la Reforma del Proceso",
            "a. La Ley 161-2009, según ",
            None,
            "enmendada, conocida como la “Ley para la Reforma del Proceso",
        ),
        ("file”).", None, ").   ", "file”"),
        (
            "8) la profundidad de atascado (“stemming”)",
            "8) la profundidad de atascado ",
            "mming”)",
            "(“ste",
        ),
    ],
)
def test_the_reading_of_a_run_is_what_lies_between_the_printed_text_beside_it(
    line, before, after, reading
):
    assert reading_between(line, before, after) == reading


def test_the_block_marker_the_model_puts_before_a_line_is_not_part_of_the_reading():
    line = "<|det|>text [39, 320, 961, 680]<|/det|>semilla (“seed file”)."

    assert reading_between(line, "semilla (", ").   ") == "“seed file”"


def test_straight_quotes_are_read_as_the_typographic_ones_the_reglamento_prints():
    line = 'd. El espacio papel ("paper space") o el plano ("layout") debe tener'

    assert reading_between(line, ") o el plano (", ") debe tener") == "“layout”"


def test_printed_text_read_a_little_differently_still_bounds_the_run():
    """The model read the printed “Aided” as "Aidad"; the run is still what comes before."""
    line = "conocidos en inglés como “Computer-Aidad"

    assert reading_between(line, "conocidos en inglés como ", "-Aided ") == "“Computer"


def test_a_run_is_unread_when_the_printed_text_beside_it_is_not_in_the_reading():
    assert reading_between("Otra línea de la página", "semilla (", ").") == ""


def test_a_run_read_as_a_formula_is_unread():
    line = "\\[ 2 4 - 2 9 ^ {\\prime \\prime} \\] \\[ 3: 1 \\]"

    assert reading_between(line, "24-", "3:1 ") == ""


def test_each_outlined_run_is_read_from_an_image_of_its_line_and_joined_into_the_text():
    """p. 80: the PDF draws a line and a half of Sección 2.1.9.5(a) as outlines."""
    answers = iter(
        [
            "la Ley Núm. 135 de 15 de junio de 1967, según enmendada, conocida como “Ley de",
            'Certificaciones" (Ley Núm. 135), serán promovidas exclusivamente por los profesionales',
        ]
    )
    images = []

    def model(image):
        images.append(image.read_bytes())
        return next(answers)

    extraction = extract_lines(RC_PDF, 80, 80, FOOTER_TOP, UnlimitedOcr(model))

    text = "\n".join(line.text for line in extraction.lines)
    assert (
        "al amparo de lo dispuesto en\n"
        "la Ley Núm. 135 de 15 de junio de 1967, según enmendada, conocida como “Ley de\n"
        "Certificaciones” (Ley Núm. 135), serán promovidas exclusivamente por los profesionales\n"
        "designados en dicha Ley."
    ) in text
    assert GAP not in text
    assert extraction.ocr_engine == "unlimited-ocr"
    assert len(images) == 2 and all(image.startswith(b"\x89PNG") for image in images)


def test_a_run_the_model_does_not_read_stays_marked():
    extraction = extract_lines(RC_PDF, 80, 80, FOOTER_TOP, UnlimitedOcr(lambda image: ""))

    assert [run.recovered for run in extraction.outlined_runs] == [False, False]
