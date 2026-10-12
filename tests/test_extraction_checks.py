import pytest

from case_mgmt_ai_demo.chunks import Chunk
from case_mgmt_ai_demo.extraction_checks import check_chunks


def seccion(number: str, text: str, part: str | None = None) -> Chunk:
    regla = number.rsplit(".", 1)[0]
    return Chunk(
        section_id=f"RC-2023-E:{number}",
        doc="RC-2023-E",
        level="seccion",
        heading="REQUISITOS",
        parents=["Tomo II", "Capítulo 2.1", f"Regla {regla}"],
        page_start=77,
        page_end=77,
        text=text,
        part=part,
        request_types=["PCOC"],
    )


CLEAN = (
    "a. Toda solicitud será promovida por el dueño de la propiedad:\n"
    "1. Escritura pública.\n"
    "2. Contrato de arrendamiento\n"
    "b. Tendrá que incluirse evidencia fehaciente de la legitimación activa."
)


def checks(*chunks: Chunk) -> list[str]:
    return [flag.check for flag in check_chunks(chunks)]


def test_a_clean_section_raises_no_flag():
    assert check_chunks([seccion("2.1.9.1", CLEAN), seccion("2.1.9.2", CLEAN)]) == []


def test_a_line_cut_mid_word_and_followed_by_an_unrelated_line_is_flagged():
    garbled = (
        "a. Aquellas solicitudes presentadas al amparo de lo dispuesto en Certi\n"
        "designados en dicha Ley."
    )

    [flag] = check_chunks([seccion("2.1.9.1", garbled)])

    assert flag.check == "truncated_line"
    assert flag.section_id == "RC-2023-E:2.1.9.1"
    assert "Certi" in flag.detail and "designados" in flag.detail


def test_text_left_unread_is_flagged():
    assert checks(seccion("2.1.9.1", "a. Lo dispuesto en […] designados en dicha Ley.")) == [
        "unread_text"
    ]


@pytest.mark.parametrize(
    "remnant", ["REGLAMENTO CONJUNTO | 41", "28 | REGLAMENTO CONJUNTO", "REGLAMENTO CONJUNTO"]
)
def test_a_running_header_left_in_the_text_is_flagged(remnant):
    text = f"a. El proponente presentará todo documento requerido\n{remnant}\nb. La ciudadanía."

    assert "running_header" in checks(seccion("2.1.9.1", text))


def test_the_reglamento_named_in_a_sentence_is_not_a_running_header():
    text = "a. Se adopta el REGLAMENTO CONJUNTO para la evaluación y expedición de permisos."

    assert checks(seccion("2.1.9.1", text)) == []


def test_a_gap_in_clause_lettering_is_flagged():
    text = "a. Primer requisito del solicitante.\nb. Segundo requisito.\nd. Cuarto requisito."

    [flag] = check_chunks([seccion("2.1.9.1", text)])

    assert (flag.check, flag.detail) == ("clause_lettering_gap", "(d) comes after (b)")


def test_lettering_that_does_not_start_at_a_is_flagged():
    text = "n. Para aquellos casos en que no se pueda localizar a algún colindante."

    [flag] = check_chunks([seccion("2.1.9.1", text)])

    assert (flag.check, flag.detail) == ("clause_lettering_gap", "(n) comes first")


def test_lettering_runs_on_across_the_parts_of_a_split_section_with_or_without_ñ():
    requirement = "Requisito que el solicitante debe cumplir."
    first = seccion("2.1.9.1", f"a. {requirement}\nb. {requirement}", part="1-of-3")
    second = seccion("2.1.9.1", f"c. {requirement}\nn. {requirement}", part="2-of-3")
    third = seccion("2.1.9.1", f"o. {requirement}\np. {requirement}", part="3-of-3")
    with_ñ = seccion("2.1.9.2", f"a. {requirement}\nn. {requirement}\nñ. {requirement}")

    flags = check_chunks([first, second, third, with_ñ])

    assert [(flag.section_id, flag.detail) for flag in flags] == [
        ("RC-2023-E:2.1.9.1", "(n) comes after (c)"),
        ("RC-2023-E:2.1.9.2", "(n) comes after (a)"),
    ]


def test_a_section_numbered_out_of_sequence_within_its_regla_is_flagged():
    chunks = [seccion("2.1.9.1", CLEAN), seccion("2.1.9.2", CLEAN), seccion("2.1.9.4", CLEAN)]

    [flag] = check_chunks(chunks)

    assert flag.check == "section_out_of_sequence"
    assert flag.section_id == "RC-2023-E:2.1.9.4"
    assert flag.detail == "Sección 2.1.9.4 where 2.1.9.3 was expected"


def test_a_section_numbered_for_another_regla_is_flagged():
    misnumbered = seccion("2.1.9.1", CLEAN).model_copy(update={"section_id": "RC-2023-E:116.1.1"})

    assert checks(misnumbered) == ["section_out_of_sequence"]


def test_the_parts_of_a_split_section_count_once_in_the_sequence():
    chunks = [
        seccion("2.1.9.1", "a. Uno, que es el primer requisito de la Sección.", part="1-of-2"),
        seccion("2.1.9.1", "b. Dos, que es el segundo requisito de la Sección.", part="2-of-2"),
        seccion("2.1.9.2", CLEAN),
    ]

    assert "section_out_of_sequence" not in checks(*chunks)


def test_an_empty_or_very_short_chunk_is_flagged():
    assert checks(seccion("2.1.9.1", "Reservada.")) == ["short_chunk"]


def test_secciones_directly_under_a_capitulo_are_not_checked_for_sequence():
    loose = seccion("2.3.5", CLEAN).model_copy(update={"parents": ["Tomo II", "Capítulo 2.3"]})

    assert check_chunks([loose]) == []
