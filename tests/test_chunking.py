import pytest
from pydantic import ValidationError

from case_mgmt_ai_demo.chunking import IngestionConfig, chunk_lines, load_ingestion_config
from case_mgmt_ai_demo.chunks import split_clauses
from case_mgmt_ai_demo.extraction import Line

CONFIG = IngestionConfig(
    doc="RC-2023-E",
    first_page=1,
    last_page=99,
    footer_top=738,
    max_chunk_chars=6000,
    request_types={"default": ["PCOC", "PERMISO_UNICO"], "3.7": ["PERMISO_UNICO"]},
)

MARGIN, BODY, HANGING, SUB_ITEM = 54.0, 144.0, 162.0, 180.0
FULL_WIDTH = 560.0
PITCH = 13.4


class Page:
    """Builds the lines of a page top to bottom, the way the Reglamento lays them out."""

    def __init__(self, number: int = 77):
        self.number = number
        self.y = 72.0
        self.lines: list[Line] = []

    def heading(self, text: str) -> "Page":
        self.y += PITCH  # headings stand apart from the text above and below
        self._add(text, MARGIN, 300.0, bold=True)
        self.y += PITCH
        return self

    def title_wrap(self, text: str) -> "Page":
        self.y -= PITCH
        self._add(text, BODY, 300.0, bold=True)
        self.y += PITCH
        return self

    def text(self, text: str, x0: float = BODY, x1: float = FULL_WIDTH) -> "Page":
        self._add(text, x0, x1)
        return self

    def cell(self, number: int, *texts: str) -> "Page":
        for text in texts:
            self._add(text, BODY, 400.0, cell=number)
        return self

    def _add(
        self, text: str, x0: float, x1: float, bold: bool = False, cell: int | None = None
    ) -> None:
        self.lines.append(Line(self.number, x0, self.y, x1, text, bold, cell))
        self.y += PITCH


def one_section(page: Page) -> Page:
    return (
        page.heading("TOMO II PROCEDIMIENTOS ADMINISTRATIVOS")
        .heading("CAPÍTULO 2.1 DISPOSICIONES GENERALES")
        .heading("REGLA 2.1.9 PRESENTACIÓN DE SOLICITUDES")
        .heading("SECCIÓN 2.1.9.5 LEGITIMACIÓN ACTIVA (STANDING)")
    )


def test_a_seccion_becomes_a_chunk_with_its_tomo_capitulo_and_regla_as_parents():
    page = one_section(Page(80))
    page.text("a. Toda solicitud será promovida por el dueño.", x1=400.0)

    [chunk] = chunk_lines(page.lines, CONFIG).chunks

    assert chunk.section_id == "RC-2023-E:2.1.9.5"
    assert (chunk.doc, chunk.level) == ("RC-2023-E", "seccion")
    assert chunk.heading == "LEGITIMACIÓN ACTIVA (STANDING)"
    assert chunk.parents == ["Tomo II", "Capítulo 2.1", "Regla 2.1.9"]
    assert (chunk.page_start, chunk.page_end, chunk.part) == (80, 80, None)
    assert chunk.text == "a. Toda solicitud será promovida por el dueño."


def test_wrapped_lines_are_joined_and_each_list_item_keeps_its_own_line():
    page = one_section(Page())
    page.text("a. Tendrá que incluirse evidencia fehaciente de que el proponente es dueño,")
    page.text("mediante copia simple de uno o más de estos documentos:", HANGING, 420.0)
    page.text("1. Escritura pública.", HANGING, 260.0)
    page.text("2. De ser una persona jurídica, se presentará documentación que acredite", HANGING)
    page.text("su existencia.", SUB_ITEM, 240.0)
    page.text("b. Cuando se trate de más de un propietario, todos autorizan.", x1=450.0)

    [chunk] = chunk_lines(page.lines, CONFIG).chunks

    assert chunk.text == (
        "a. Tendrá que incluirse evidencia fehaciente de que el proponente es dueño, "
        "mediante copia simple de uno o más de estos documentos:\n"
        "1. Escritura pública.\n"
        "2. De ser una persona jurídica, se presentará documentación que acredite "
        "su existencia.\n"
        "b. Cuando se trate de más de un propietario, todos autorizan."
    )
    assert [clause.ref[-3:] for clause in split_clauses(chunk)] == ["(a)", "(b)"]


def test_unlettered_paragraphs_are_kept_apart():
    page = one_section(Page())
    page.text("Los usos a permitirse serán compatibles con el distrito y con")
    page.text("las disposiciones de esta Sección.", x1=300.0)
    page.text("Toda nueva construcción deberá observar los parámetros.", x1=420.0)

    [chunk] = chunk_lines(page.lines, CONFIG).chunks

    assert chunk.text.split("\n") == [
        "Los usos a permitirse serán compatibles con el distrito y con las disposiciones "
        "de esta Sección.",
        "Toda nueva construcción deberá observar los parámetros.",
    ]


def test_a_lettered_item_under_another_item_is_indented_so_it_is_not_read_as_a_clause():
    page = one_section(Page())
    page.text("e. Solicitud de Copia de Documentos o Expediente:", x1=400.0)
    page.text("a. El SUI mantendrá un expediente oficial.", HANGING, 400.0)
    page.text("b. Cualquier persona interesada puede obtener copia.", HANGING, 420.0)
    page.text("f. Otros servicios.", x1=250.0)

    [chunk] = chunk_lines(page.lines, CONFIG).chunks

    assert chunk.text.split("\n")[1] == "  a. El SUI mantendrá un expediente oficial."
    assert [clause.ref[-3:] for clause in split_clauses(chunk)] == ["(e)", "(f)"]


def test_a_section_running_onto_the_next_page_spans_both_pages():
    first, second = one_section(Page(80)), Page(81)
    first.text("a. Toda solicitud será promovida por el dueño, optante o arrendatario de la")
    second.text("propiedad con autorización.", HANGING, 300.0)
    second.text("b. Tendrá que incluirse evidencia.", x1=330.0)

    [chunk] = chunk_lines(first.lines + second.lines, CONFIG).chunks

    assert (chunk.page_start, chunk.page_end) == (80, 81)
    assert chunk.text.startswith(
        "a. Toda solicitud será promovida por el dueño, optante o arrendatario de la "
        "propiedad con autorización.\n"
    )


def test_page_start_is_the_page_of_the_heading_even_when_the_text_starts_on_the_next():
    first, second = one_section(Page(80)), Page(81)
    second.text("a. Toda solicitud será promovida por el dueño.", x1=400.0)

    [chunk] = chunk_lines(first.lines + second.lines, CONFIG).chunks

    assert (chunk.page_start, chunk.page_end) == (80, 81)


def test_a_regla_without_secciones_is_a_chunk_at_regla_level():
    page = Page(105)
    page.heading("TOMO II PROCEDIMIENTOS").heading("CAPITULO 2.2 CRITERIOS DE EVALUACIÓN")
    page.heading("REGLA 2.2.6 PRÓRROGA A LA VIGENCIA DE PERMISO DE CONSTRUCCIÓN")
    page.text("a. Solamente se concederá un máximo de dos (2) prórrogas.", x1=450.0)

    result = chunk_lines(page.lines, CONFIG)

    [chunk] = result.chunks
    assert (chunk.section_id, chunk.level) == ("RC-2023-E:2.2.6", "regla")
    assert chunk.parents == ["Tomo II", "Capítulo 2.2"]
    assert [(s.section_id, s.level) for s in result.skipped] == [
        ("RC-2023-E:Tomo-II", "tomo"),
        ("RC-2023-E:2.2", "capitulo"),
    ]


def test_text_under_a_regla_before_its_first_seccion_is_its_own_chunk():
    page = Page()
    page.heading("REGLA 3.7.1 PERMISO ÚNICO")
    page.text("Esta Regla establece el Permiso Único.", x1=380.0)
    page.heading("SECCIÓN 3.7.1.1 DISPOSICIONES GENERALES")
    page.text("a. El Permiso Único consolida trámites.", x1=380.0)

    chunks = chunk_lines(page.lines, CONFIG).chunks

    assert [(c.section_id, c.level, c.parents) for c in chunks] == [
        ("RC-2023-E:3.7.1", "regla", []),
        ("RC-2023-E:3.7.1.1", "seccion", ["Regla 3.7.1"]),
    ]


def test_a_new_capitulo_or_tomo_closes_the_levels_below_it():
    page = Page()
    page.heading("TOMO II PROCEDIMIENTOS").heading("CAPÍTULO 2.1 DISPOSICIONES")
    page.heading("REGLA 2.1.1 PROPÓSITO").text("Texto de la Regla 2.1.1.", x1=300.0)
    page.heading("CAPÍTULO 2.2 CRITERIOS").text("Texto del Capítulo 2.2.", x1=300.0)
    page.heading("TOMO III PERMISOS").heading("CAPÍTULO 3.1 DISPOSICIONES")
    page.heading("REGLA 3.1.1 ALCANCE").text("Texto de la Regla 3.1.1.", x1=300.0)

    chunks = chunk_lines(page.lines, CONFIG).chunks

    assert [(c.section_id, c.level, c.parents) for c in chunks] == [
        ("RC-2023-E:2.1.1", "regla", ["Tomo II", "Capítulo 2.1"]),
        ("RC-2023-E:2.2", "capitulo", ["Tomo II"]),
        ("RC-2023-E:3.1.1", "regla", ["Tomo III", "Capítulo 3.1"]),
    ]


def test_a_long_title_that_wraps_is_part_of_the_heading_not_of_the_text():
    page = Page()
    page.heading("REGLA 2.1.4 VIGENCIAS DE CONSULTAS DE UBICACIÓN Y TRANSACCIONES")
    page.title_wrap("VIGENTES AL MOMENTO DE LA LEY 142-2012")
    page.text("a. Las consultas se prolongarán.", x1=330.0)

    [chunk] = chunk_lines(page.lines, CONFIG).chunks

    assert chunk.heading == (
        "VIGENCIAS DE CONSULTAS DE UBICACIÓN Y TRANSACCIONES VIGENTES AL MOMENTO DE LA LEY 142-2012"
    )
    assert chunk.text == "a. Las consultas se prolongarán."


def test_headings_misprinted_in_the_reglamento_are_still_headings():
    page = Page()
    page.heading("CAPITULO 1.2 JUNTA DE PLANIFICACIÓN").text("Texto del capítulo.", x1=300.0)
    page.heading("REGLA 6.1.28 DISTRITO").heading("SECCIÓN 6.1. 28.2 USOS")
    page.text("Los usos a permitirse.", x1=300.0)
    page.heading("SECCCIÓN 8.7.4.7 OTROS RÓTULOS Y ANUNCIOS").text("Los rótulos.", x1=300.0)

    chunks = chunk_lines(page.lines, CONFIG).chunks

    assert [(c.section_id, c.heading) for c in chunks] == [
        ("RC-2023-E:1.2", "JUNTA DE PLANIFICACIÓN"),
        ("RC-2023-E:6.1.28.2", "USOS"),
        ("RC-2023-E:8.7.4.7", "OTROS RÓTULOS Y ANUNCIOS"),
    ]


def test_a_reference_to_a_seccion_inside_the_text_is_not_a_heading():
    page = one_section(Page())
    page.text("a. Se cumplirá con lo dispuesto en la")
    page.text("SECCIÓN 2.1.9.7 sobre colindantes.", HANGING, 380.0)

    [chunk] = chunk_lines(page.lines, CONFIG).chunks

    assert chunk.text == "a. Se cumplirá con lo dispuesto en la SECCIÓN 2.1.9.7 sobre colindantes."


def test_a_number_the_reglamento_uses_twice_gets_a_bis_suffix_so_section_ids_stay_unique():
    page = Page(469)
    page.heading("REGLA 6.1.16 DISTRITO I-P - INDUSTRIAL PESADO")
    page.heading("SECCIÓN 6.1.16.1 PROPÓSITO").text("Clasifica áreas industriales.", x1=330.0)
    page.heading("REGLA 6.1.16 DISTRITO ARD: ÁREA RURAL DESARROLLADA")
    page.heading("SECCIÓN 6.1.16.1 PROPÓSITO").text("Facilita el control del área.", x1=330.0)

    result = chunk_lines(page.lines, CONFIG)

    assert [(c.section_id, c.parents) for c in result.chunks] == [
        ("RC-2023-E:6.1.16.1", ["Regla 6.1.16"]),
        ("RC-2023-E:6.1.16.1-bis", ["Regla 6.1.16-bis"]),
    ]
    assert result.repeated == ["RC-2023-E:6.1.16-bis", "RC-2023-E:6.1.16.1-bis"]


def test_a_long_section_is_split_where_a_clause_starts_into_parts_sharing_its_section_id():
    first, second = one_section(Page(77)), Page(78)
    for letter in "abc":
        first.text(f"{letter}. " + "requisito " * 150, x1=400.0)
    for letter in "def":
        second.text(f"{letter}. " + "requisito " * 150, x1=400.0)
    config = CONFIG.model_copy(update={"max_chunk_chars": 5000})

    parts = chunk_lines(first.lines + second.lines, config).chunks

    assert [(p.section_id, p.part) for p in parts] == [
        ("RC-2023-E:2.1.9.5", "1-of-2"),
        ("RC-2023-E:2.1.9.5", "2-of-2"),
    ]
    assert [[clause.ref[-3:] for clause in split_clauses(p)] for p in parts] == [
        ["(a)", "(b)", "(c)"],
        ["(d)", "(e)", "(f)"],
    ]
    assert all(len(part.text) <= 5000 for part in parts)


def test_each_part_of_a_split_section_has_the_pages_of_its_own_text():
    first, second = one_section(Page(77)), Page(79)
    first.text("a. " + "requisito " * 300, x1=400.0)
    second.text("b. " + "requisito " * 300, x1=400.0)
    config = CONFIG.model_copy(update={"max_chunk_chars": 4000})

    parts = chunk_lines(first.lines + second.lines, config).chunks

    assert [(p.part, p.page_start, p.page_end) for p in parts] == [
        ("1-of-2", 77, 77),
        ("2-of-2", 79, 79),
    ]


def test_a_clause_too_long_for_one_part_is_cut_between_its_items_then_between_sentences():
    page = one_section(Page())
    page.text("a. Requisitos:", x1=250.0)
    page.text("1. " + "Primera oración del requisito. " * 60, HANGING, 400.0)
    page.text("2. " + "documento " * 100, HANGING, 400.0)
    config = CONFIG.model_copy(update={"max_chunk_chars": 1200})

    parts = chunk_lines(page.lines, config).chunks

    assert len(parts) > 2
    assert all(len(part.text) <= 1200 for part in parts)
    assert parts[0].text.startswith("a. Requisitos:\n1. Primera oración del requisito.")
    assert all(part.text.endswith(("requisito.", "documento")) for part in parts)
    rejoined = " ".join(part.text.replace("\n", " ") for part in parts)
    assert rejoined.count("Primera oración") == 60 and rejoined.count("documento") == 100


def test_request_types_come_from_the_longest_matching_number_prefix():
    page = Page()
    page.heading("SECCIÓN 3.7.1.2 REQUISITOS").text("Requisitos del Permiso Único.", x1=300.0)
    page.heading("SECCIÓN 3.70.1.1 OTRA").text("No es del Capítulo 3.7.", x1=300.0)
    page.heading("SECCIÓN 2.1.9.3 VALOR").text("Valor estimado de la obra.", x1=300.0)
    config = CONFIG.model_copy(
        update={
            "request_types": {
                "default": ["PCOC", "PERMISO_UNICO"],
                "3.7": ["PERMISO_UNICO"],
                "2": ["PERMISO_UNICO"],
                "2.1.9.3": ["PCOC"],
            }
        }
    )

    chunks = chunk_lines(page.lines, config).chunks

    assert [chunk.request_types for chunk in chunks] == [
        ["PERMISO_UNICO"],
        ["PCOC", "PERMISO_UNICO"],
        ["PCOC"],
    ]


def test_the_rc_2023_e_configuration_skips_the_front_matter_the_contents_and_the_glossary():
    config = load_ingestion_config()

    assert config.doc == "RC-2023-E"
    assert (config.first_page, config.last_page) == (53, 824)
    assert config.request_types["default"] == ["PCOC", "PERMISO_UNICO"]


def test_request_type_rules_must_say_what_untagged_sections_get():
    with pytest.raises(ValidationError, match="default"):
        IngestionConfig.model_validate(
            CONFIG.model_dump() | {"request_types": {"3.7": ["PERMISO_UNICO"]}}
        )


def test_a_number_used_many_times_keeps_getting_distinct_ids():
    page = Page()
    for _ in range(5):
        page.heading("SECCIÓN 6.1.9.2 USOS").text("Los usos a permitirse.", x1=300.0)

    chunks = chunk_lines(page.lines, CONFIG).chunks

    assert [chunk.section_id.removeprefix("RC-2023-E:6.1.9.2") for chunk in chunks] == [
        "",
        "-bis",
        "-bis2",
        "-bis3",
        "-bis4",
    ]


def test_each_table_cell_is_its_own_paragraph_and_its_lettered_items_are_not_clauses():
    page = one_section(Page(417))
    page.text("Toda nueva construcción deberá observar los siguientes parámetros:", x1=500.0)
    page.cell(0, "Cabida mínima de", "solar").cell(1, "275 metros cuadrados")
    page.cell(2, "Densidad", "poblacional")
    page.cell(3, "a. Solares menores, una unidad de vivienda.", "b. Solares mayores, 1 UBV.")
    page.text("a. Los parámetros anteriores no admiten variación.", x1=450.0)

    [chunk] = chunk_lines(page.lines, CONFIG).chunks

    assert chunk.text.split("\n") == [
        "Toda nueva construcción deberá observar los siguientes parámetros:",
        "Cabida mínima de solar",
        "275 metros cuadrados",
        "Densidad poblacional",
        "  a. Solares menores, una unidad de vivienda.",
        "  b. Solares mayores, 1 UBV.",
        "a. Los parámetros anteriores no admiten variación.",
    ]
    assert [clause.ref for clause in split_clauses(chunk)][1:] == ["RC-2023-E:2.1.9.5(a)"]
