"""Chunking of the real COBROS OA-2022-07 PDF: the order's ordinals and Anejo A's items."""

from case_mgmt_ai_demo.chunks import Chunk, split_clauses


def section(chunking, section_id: str) -> list[Chunk]:
    return [chunk for chunk in chunking.chunks if chunk.section_id == section_id]


def test_anejo_a_item_6_holds_the_pcoc_fee_formula_intact(cobros_chunking):
    [pcoc] = section(cobros_chunking, "COBROS:AnejoA:6")

    assert pcoc.level == "anejo_item"
    assert pcoc.heading == "Permiso de Construcción (PCOC) y Permiso de Construccion Verde"
    assert pcoc.parents == ["Anejo A", "EDIFICABILIDAD"]
    assert pcoc.request_types == ["PCOC"]
    assert pcoc.text.split("\n")[:3] == [
        "a. Radicación (Cargo por servicio | Derechos a cobrarse: $ 0.35 | Concepto: Por cada "
        "$1,000 del valor de la obra o fracción. Pago mínimo de $100.00 Lo que resulte mayor",
        "b. Aranceles | Derechos a cobrarse: $5.00 | Concepto: Por cada $1,000 del valor de la "
        "obra o fracción.",
        "c. Sellos del CIAPR y CAAPPR | Derechos a cobrarse: $1.00 | Concepto: Por cada $1,000 "
        "del valor de la obra o fracción.",
    ]
    assert [clause.ref[-3:] for clause in split_clauses(pcoc)] == [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
    ]


def test_an_item_whose_rows_cross_a_page_break_keeps_both_pages(cobros_chunking):
    [pcoc] = section(cobros_chunking, "COBROS:AnejoA:6")

    # The heading and row (a) are at the foot of p. 13; rows (b) to (e) are on p. 14.
    assert (pcoc.page_start, pcoc.page_end) == (13, 14)


def test_the_order_is_chunked_by_ordinal_and_anejo_a_by_item(cobros_chunking):
    ids = [chunk.section_id for chunk in cobros_chunking.chunks if chunk.part_number == 1]

    ordinals = "PRIMERO SEGUNDO TERCERO CUARTO QUINTO SEXTO SÉPTIMO OCTAVO NOVENO".split()
    assert ids == [f"COBROS:{o}" for o in ordinals] + [f"COBROS:AnejoA:{n}" for n in range(1, 58)]
    assert cobros_chunking.skipped == []
    assert {chunk.doc for chunk in cobros_chunking.chunks} == {"COBROS-OA-2022-07"}


def test_tercero_holds_the_exemption_table_row_by_row_under_its_column_names(cobros_chunking):
    tercero = section(cobros_chunking, "COBROS:TERCERO")
    rows = [line for part in tercero for line in part.text.split("\n")]

    assert (tercero[0].level, tercero[0].heading) == ("seccion", "AJUSTES EN LOS CARGOS")
    assert (tercero[0].page_start, tercero[-1].page_end) == (5, 7)
    assert (
        "Cuerpo de Ingenieros de los Estados Unidos de América"
        " | Licencia Sanitaria y Certificación de Prevención de Incendios: No aplica"
        " | Derechos por Servicios: Exento para trámites ambientales en todas sus etapas y"
        " trámite de Recomendaciones"
        " | Derechos por aranceles: No aplica"
        " | Derechos por estampillas: No aplica"
    ) in rows
    [government] = [row for row in rows if row.startswith("Gobierno Estatal, Federal, Municipal")]
    assert "Derechos por Servicios: 25 % de los cargos y derechos aplicables" in government
    assert "Derechos por estampillas: Exentos, si las obras son llevadas a cabo" in government
    assert rows[-1].startswith("6. La Secretaria Auxiliar se reserva la discreción")


def test_a_table_row_cut_by_a_page_break_is_one_row(cobros_chunking):
    tercero = "\n".join(part.text for part in section(cobros_chunking, "COBROS:TERCERO"))
    [sanitary] = section(cobros_chunking, "COBROS:AnejoA:30")

    # pp. 5-6: the row has a rule above it on neither page.
    assert (
        "Entidades educativas que acrediten su condición educadora o se encuentre bajo las "
        "disposiciones reglamentarias del Consejo de Educación Superior de Puerto Rico. | "
    ) in tercero
    # pp. 22-23: Anejo A rules off the top of every page.
    assert (
        "h. Cementerios, crematorios, embalsamadoras y funerarias | Derechos a cobrarse: $50.00 "
        "por cada año solicitado."
    ) in sanitary.text.split("\n")


def test_a_row_without_a_letter_stays_in_the_clause_above_it(cobros_chunking):
    [fire] = section(cobros_chunking, "COBROS:AnejoA:28")

    [first, *_] = split_clauses(fire)

    assert first.ref == "COBROS:AnejoA:28(a)"
    assert first.text.split("\n")[:2] == [
        "a. Certificación Prevención de Incendios | Derechos a cobrarse: $60.00 por cada año "
        "solicitado. | Comentarios: Hasta 1,000 p.c. | Concepto: Por local o estructura",
        "Derechos a cobrarse: $70.00 por cada año solicitado. | Comentarios: Desde 1,001 p.c. "
        "hasta 3,000 p.c. | Concepto: Por local o estructura",
    ]


def test_an_item_of_a_single_row_is_that_row(cobros_chunking):
    [revision] = section(cobros_chunking, "COBROS:AnejoA:46")

    assert revision.heading == "Revisión a una Determinación"
    assert revision.parents == ["Anejo A", "DIVISIÓN DE REVISIONES ADMINISTRATIVAS"]
    assert revision.text == "Revisión a una Determinación | Derechos a cobrarse: $25.00"


def test_a_footnote_follows_the_row_that_refers_to_it(cobros_chunking):
    [other_uses] = section(cobros_chunking, "COBROS:AnejoA:42")
    rows = other_uses.text.split("\n")

    late = rows.index(
        "c. Renovación tardía Permiso Único (Uso)1 | Derechos a cobrarse: $75.00 | Concepto: "
        "Más Costo de licencias y certificaciones"
    )
    assert rows[late + 1] == "1 Se refiere a las solicitudes radicadas posterior al vencimiento."
    assert rows[late + 2].startswith("d. Enmienda Permiso Único")


def test_no_chunk_holds_the_running_header_the_address_the_recitals_or_the_signature(
    cobros_chunking,
):
    text = "\n".join(chunk.text for chunk in cobros_chunking.chunks)

    for remnant in (
        "Orden Administrativa OGPe 2022-07",
        "Para establecer el cobro por solicitudes",
        "Roosevelt",
        "PO Box 41179",
        "POR CUANTO",
        "POR TODO LO CUAL",
        "Secretario Auxiliar\n",
        "# | Producto",
    ):
        assert remnant not in text
    assert section(cobros_chunking, "COBROS:NOVENO")[0].text.endswith("Orden Administrativa.")


def test_request_types_narrow_the_items_the_fee_check_charges(cobros_chunking):
    types = {chunk.section_id: chunk.request_types for chunk in cobros_chunking.chunks}

    assert types["COBROS:AnejoA:6"] == ["PCOC"]
    assert types["COBROS:AnejoA:28"] == types["COBROS:AnejoA:41"] == ["PERMISO_UNICO"]
    assert types["COBROS:AnejoA:1"] == types["COBROS:TERCERO"] == ["PCOC", "PERMISO_UNICO"]


def test_a_division_named_over_two_rows_gives_both_as_parents(cobros_chunking):
    [uses] = section(cobros_chunking, "COBROS:AnejoA:41")
    [temporary] = section(cobros_chunking, "COBROS:AnejoA:43")
    [review] = section(cobros_chunking, "COBROS:AnejoA:46")

    assert uses.parents == temporary.parents == ["Anejo A", "DIVISION DE USOS", "Usos"]
    assert review.parents == ["Anejo A", "DIVISIÓN DE REVISIONES ADMINISTRATIVAS"]
