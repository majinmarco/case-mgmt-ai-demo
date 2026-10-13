import re

import pytest

from case_mgmt_ai_demo.extraction import GAP, Tesseract, extract_lines, find_tessdata
from conftest import RC_PDF

FOOTER_TOP = 738


def page_text(extraction, page: int) -> str:
    return "\n".join(line.text for line in extraction.lines if line.page == page)


def test_lines_carry_their_physical_page_and_come_in_reading_order():
    extraction = extract_lines(RC_PDF, 80, 81, FOOTER_TOP)

    assert {line.page for line in extraction.lines} == {80, 81}
    first = extraction.lines[0]
    assert (first.page, first.text) == (80, "SECCIÓN 2.1.9.3 VALOR ESTIMADO DE LA OBRA")
    assert first.bold and first.x0 < 60


def test_a_list_marker_and_its_text_printed_apart_are_one_line():
    extraction = extract_lines(RC_PDF, 77, 77, FOOTER_TOP)

    assert extraction.lines[0].text == "e. Solicitud de Copia de Documentos o Expediente:"


def test_the_running_header_is_stripped(rc_extraction):
    headers = [
        line for line in rc_extraction.lines if re.search(r"REGLAMENTO CONJUNTO\s*\|", line.text)
    ]

    assert headers == []
    assert "| 25" not in page_text(rc_extraction, 77)


def test_text_drawn_as_outlines_is_marked_where_it_is_missing_when_not_read():
    """The p. 80 gap in Sección 2.1.9.5(a): the PDF draws a line and a half as outlines."""
    extraction = extract_lines(RC_PDF, 80, 80, FOOTER_TOP)

    text = page_text(extraction, 80)
    assert f"al amparo de lo dispuesto en\n{GAP}\nCerti {GAP}\ndesignados en dicha Ley." in text
    assert not extraction.ocr
    assert [(run.page, run.recovered) for run in extraction.outlined_runs] == [(80, False)] * 2


@pytest.mark.skipif(find_tessdata() is None, reason="needs Tesseract's Spanish model")
def test_text_drawn_as_outlines_is_read_with_ocr_and_joined_to_the_word_it_continues():
    tessdata = find_tessdata()
    assert tessdata is not None
    extraction = extract_lines(RC_PDF, 80, 80, FOOTER_TOP, Tesseract(tessdata))

    text = page_text(extraction, 80)
    assert (
        "al amparo de lo dispuesto en\n"
        "la Ley Núm. 135 de 15 de junio de 1967, según enmendada, conocida como “Ley de\n"
        "Certificaciones” (Ley Núm. 135), serán promovidas exclusivamente por los profesionales\n"
        "designados en dicha Ley."
    ) in text
    assert GAP not in text
    assert all(run.recovered for run in extraction.outlined_runs)


def test_blank_pages_and_tomo_cover_pages_give_no_lines(rc_extraction):
    """A cover (p. 173, "TOMO III") has no text, only outlines; they are not marked as gaps."""
    assert not [line for line in rc_extraction.lines if line.page in (53, 69, 172, 173)]
    assert not [run for run in rc_extraction.outlined_runs if run.page in (53, 173)]


def test_a_table_is_read_cell_by_cell_not_straight_across():
    """Tabla 6.34 on p. 417: a label cell centred beside several lines of description."""
    extraction = extract_lines(RC_PDF, 417, 417, FOOTER_TOP)

    text = page_text(extraction, 417)
    assert (
        "Cabida mínima de\nsolar\n275 metros cuadrados\nAncho mínimo de\nsolar\n11 metros" in text
    )
    assert (
        "Densidad\npoblacional\n"
        "a. Solares menores a doscientos setenta y cinco (275) metros\n"
        "cuadrados, una unidad de vivienda.\n"
        "b. Solares de doscientos setenta y cinco (275) metros cuadrados o"
    ) in text
    label, description = (
        next(line for line in extraction.lines if line.text == wanted)
        for wanted in ("Cabida mínima de", "275 metros cuadrados")
    )
    assert label.cell is not None and description.cell == label.cell + 1


def test_text_outside_a_table_is_in_no_cell():
    extraction = extract_lines(RC_PDF, 80, 80, FOOTER_TOP)

    assert {line.cell for line in extraction.lines} == {None}
