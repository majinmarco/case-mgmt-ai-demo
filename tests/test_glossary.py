"""Chunking of the glossary (Tomo XII): one chunk per term."""

from case_mgmt_ai_demo.chunking import IngestionConfig, chunk_glossary
from case_mgmt_ai_demo.chunks import Chunk
from case_mgmt_ai_demo.extraction import Line
from case_mgmt_ai_demo.index import chunk_id

CONFIG = IngestionConfig(
    doc="RC-2023-E",
    first_page=1,
    last_page=99,
    footer_top=738,
    max_chunk_chars=6000,
    request_types={"default": ["PCOC", "PERMISO_UNICO"]},
)
MARGIN, HANGING = 54.0, 81.0


def term(glossary, section_id: str) -> Chunk:
    [chunk] = [chunk for chunk in glossary.chunks if chunk.section_id == section_id]
    return chunk


def line(
    y: float, text: str, x0: float = HANGING, bold: str | None = None, page: int = 829
) -> Line:
    """`bold` is the bold text the line opens with; a line that is all bold names itself."""
    spans = (bold,) if bold else ()
    return Line(page, x0, y, 557.0, text, bold=bold == text, bold_spans=spans)


def test_each_term_is_a_chunk_named_after_the_term_and_holding_its_definition(glossary_chunking):
    haulage = term(glossary_chunking, "RC-2023-E:Glosario:A.1")

    assert (haulage.level, haulage.heading) == ("glossary_term", "Acarreo")
    assert haulage.parents == ["Tomo XII", "Glosario de términos"]
    assert haulage.request_types == ["PCOC", "PERMISO_UNICO"]
    assert haulage.text.startswith(
        "Acción de cargar y transportar de un lugar a otro los materiales"
    )
    assert haulage.text.endswith("de materiales y de materia prima de otros procesos.")
    assert term(glossary_chunking, "RC-2023-E:Glosario:A.2").text == (
        "Cualidad de poder llegar a un destino físico."
    )


def test_every_term_of_every_letter_is_chunked_once(glossary_chunking):
    ids = [chunk_id(chunk) for chunk in glossary_chunking.chunks]

    assert len(ids) == len(set(ids)) == 1431
    assert glossary_chunking.skipped == []
    assert ids[0] == "RC-2023-E:Glosario:A.1"
    assert ids[-1] == "RC-2023-E:Glosario:Z.21"
    assert {chunk.level for chunk in glossary_chunking.chunks} == {"glossary_term"}


def test_page_start_is_the_physical_page_holding_the_term(glossary_chunking):
    assert term(glossary_chunking, "RC-2023-E:Glosario:A.1").page_start == 829
    proposed = term(glossary_chunking, "RC-2023-E:Glosario:A.13")  # runs over onto p. 830
    assert (proposed.page_start, proposed.page_end) == (829, 830)
    assert term(glossary_chunking, "RC-2023-E:Glosario:Z.21").page_start == 955


def test_a_dash_between_a_term_and_its_definition_belongs_to_neither(glossary_chunking):
    vehicular = term(glossary_chunking, "RC-2023-E:Glosario:A.14")

    assert vehicular.heading == "Acceso Vehicular"
    assert vehicular.text.startswith("Vía o carretera pública que permite")


def test_a_term_too_long_for_one_line_keeps_its_whole_name(glossary_chunking):
    fema = term(glossary_chunking, "RC-2023-E:Glosario:A.38")

    assert fema.heading == (
        "Agencia Federal para el Manejo de Emergencias (Federal Emergency Management Agency, FEMA)"
    )
    assert fema.text.startswith("Agencia Federal para el Manejo de Emergencias del Departamento")


def test_a_number_the_glossary_uses_twice_under_a_letter_keeps_both_terms(glossary_chunking):
    assert term(glossary_chunking, "RC-2023-E:Glosario:D.2").heading == "Datum"
    assert term(glossary_chunking, "RC-2023-E:Glosario:D.2-bis").heading == "Declaración"
    assert "RC-2023-E:Glosario:D.2-bis" in glossary_chunking.repeated


def test_the_analysis_that_follows_the_terms_is_not_part_of_the_glossary(glossary_chunking):
    last = glossary_chunking.chunks[-1]

    assert last.text == "Instrumento para designar usos de terrenos, será sinónimo de Calificación."
    assert max(chunk.page_end for chunk in glossary_chunking.chunks) == 955


def test_terms_are_numbered_under_their_letter():
    lines = [
        line(72, "III. GLOSARIO DE TÉRMINOS", MARGIN, bold="III. GLOSARIO DE TÉRMINOS"),
        line(97, "A.", MARGIN, bold="A."),
        line(122, "1. Acera Porción de una vía pública destinada", MARGIN, bold="Acera"),
        line(136, "al paso de peatones."),
        line(161, "B.", MARGIN, bold="B."),
        line(186, "1. Balcón - Plataforma que sobresale de la fachada.", MARGIN, bold="Balcón -"),
    ]

    chunks = chunk_glossary(lines, CONFIG).chunks

    assert [(chunk.section_id, chunk.heading, chunk.text) for chunk in chunks] == [
        (
            "RC-2023-E:Glosario:A.1",
            "Acera",
            "Porción de una vía pública destinada al paso de peatones.",
        ),
        ("RC-2023-E:Glosario:B.1", "Balcón", "Plataforma que sobresale de la fachada."),
    ]


def test_a_list_inside_a_definition_stays_in_its_term():
    lines = [
        line(97, "C.", MARGIN, bold="C."),
        line(122, "1. Categoría Se clasifica así:", MARGIN, bold="Categoría"),
        line(136, "1. Primera clase.", 99.0),
        line(150, "2. Segunda clase.", 99.0),
        line(175, "2. Cabida Área de un solar.", MARGIN, bold="Cabida"),
    ]

    chunks = chunk_glossary(lines, CONFIG).chunks

    assert [chunk.heading for chunk in chunks] == ["Categoría", "Cabida"]
    assert chunks[0].text == "Se clasifica así:\n1. Primera clase.\n2. Segunda clase."


def test_a_term_without_a_definition_is_reported_not_chunked():
    lines = [
        line(97, "N.", MARGIN, bold="N."),
        line(122, "1. N.S.F.", MARGIN, bold="N.S.F."),
        line(147, "2. Nivel Altura de un piso.", MARGIN, bold="Nivel"),
    ]

    glossary = chunk_glossary(lines, CONFIG)

    assert [chunk.section_id for chunk in glossary.chunks] == ["RC-2023-E:Glosario:N.2"]
    assert [(s.section_id, s.heading) for s in glossary.skipped] == [
        ("RC-2023-E:Glosario:N.1", "N.S.F.")
    ]


def test_a_term_whose_number_is_indented_is_still_its_own_term(glossary_chunking):
    # D.30 is printed further in (x=72) than the terms around it (x=54).
    medical = term(glossary_chunking, "RC-2023-E:Glosario:D.30")

    assert (medical.heading, medical.page_start) == ("Desperdicios Médicos Regulados", 864)
    assert "Desperdicios Médicos" not in term(glossary_chunking, "RC-2023-E:Glosario:D.29").text


def test_a_closing_parenthesis_set_in_regular_type_stays_with_the_term():
    lines = [
        line(97, "C.", MARGIN, bold="C."),
        line(122, "1. Centro (CRMR) - Lugar de recogido.", MARGIN, bold="Centro (CRMR"),
    ]

    [chunk] = chunk_glossary(lines, CONFIG).chunks

    assert (chunk.heading, chunk.text) == ("Centro (CRMR)", "Lugar de recogido.")


def test_bold_that_stops_inside_a_word_does_not_name_the_term():
    lines = [
        line(97, "C.", MARGIN, bold="C."),
        line(122, "1. Certificación de Uso Documento expedido.", MARGIN, bold="Certi"),
    ]

    [chunk] = chunk_glossary(lines, CONFIG).chunks

    assert (chunk.heading, chunk.text) == ("", "Certificación de Uso Documento expedido.")
