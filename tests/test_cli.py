import json
from collections.abc import Mapping
from typing import Any

import pytest

from case_mgmt_ai_demo import main
from case_mgmt_ai_demo.fee_check import check_fee, load_fee_rules
from case_mgmt_ai_demo.permit_request import parse_permit_request
from case_mgmt_ai_demo.review import (
    ClauseSelection,
    DroppedNote,
    InvalidPermitRequest,
    Note,
    ModelFailure,
    Recommendation,
    RequirementAssessment,
    SectionRelevance,
)
from test_review import GOOD_PCOC, GOOD_PERMISO_UNICO, PCOC_PARAMETERS, PCOC_STANDING, PU_USE


class FakePipeline:
    def __init__(self, outcome: Recommendation | Exception):
        self.outcome = outcome
        self.reviewed: list[Mapping[str, Any]] = []

    def review(self, raw: Mapping[str, Any]) -> Recommendation:
        self.reviewed.append(raw)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def criterion_requirement(criterion, probability) -> RequirementAssessment:
    return RequirementAssessment(
        id=criterion.id,
        text_es=criterion.text_es,
        clause_refs=criterion.pinned,
        probability=probability,
        criterion=criterion,
    )


def recommendation(**overrides) -> Recommendation:
    fields = dict(
        request=parse_permit_request(GOOD_PCOC),
        approval_score=0.86,
        verdict="approve",
        borderline=False,
        requirements=[
            criterion_requirement(PCOC_STANDING, 0.91),
            criterion_requirement(PCOC_PARAMETERS, 0.97),
            RequirementAssessment(
                id="seccion_2_1_9_2",
                text_es="Requisitos aplicables de la Sección 2.1.9.2 (REQUISITOS DE PRESENTACIÓN)",
                clause_refs=["RC-2023-E:2.1.9.2(k)"],
                probability=0.86,
            ),
        ],
        notes=[
            Note(
                text="La solicitante es dueña y presenta copia de la escritura pública.",
                section_ids=["RC-2023-E:2.1.9.5(b)"],
                curable=False,
            ),
            Note(
                text="Falta la determinación de cumplimiento ambiental.",
                section_ids=["RC-2023-E:2.1.9.2(g)"],
                curable=True,
            ),
        ],
        retrieved_section_ids=["RC-2023-E:2.1.9.2", "RC-2023-E:2.1.9.5", "RC-2023-E:2.2.6"],
        section_relevance=[
            SectionRelevance(section_id="RC-2023-E:2.1.9.2", applies=True, reason="Aplica."),
            SectionRelevance(section_id="RC-2023-E:2.1.9.5", applies=True, reason="Aplica."),
            SectionRelevance(
                section_id="RC-2023-E:2.2.6",
                applies=False,
                reason="No se solicita una prórroga del permiso.",
            ),
        ],
        selection=[
            ClauseSelection(
                ref="RC-2023-E:2.1.9.2(a)",
                category="tramite_del_sistema",
                reason="La presentación en el SUI la gestiona el sistema.",
            ),
            ClauseSelection(
                ref="RC-2023-E:2.1.9.2(k)",
                category="aplica",
                reason="Los planos deben estar sellados y certificados.",
            ),
            ClauseSelection(
                ref="RC-2023-E:2.1.9.2(l)",
                category="no_aplica",
                reason="Solo para asuntos discrecionales.",
            ),
        ],
        unclassified_clause_refs=[],
        unknown_clause_refs=[],
    )
    return Recommendation.model_validate(fields | overrides)


@pytest.fixture
def request_file(tmp_path):
    path = tmp_path / "solicitud.json"
    path.write_text(json.dumps(GOOD_PCOC, ensure_ascii=False), encoding="utf-8")
    return path


def test_review_prints_score_verdict_and_cited_notes_in_spanish(request_file, capsys):
    pipeline = FakePipeline(recommendation())

    exit_code = main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert pipeline.reviewed == [GOOD_PCOC]
    assert "María Isabel Rivera Ortiz" in out
    assert "Puntuación de aprobación: 0.86" in out
    assert "Veredicto: Aprobar" in out
    assert "1. La solicitante es dueña" in out
    assert "[RC-2023-E:2.1.9.5(b)]" in out
    assert (
        "2. Falta la determinación de cumplimiento ambiental. [RC-2023-E:2.1.9.2(g)] (subsanable)"
    ) in out
    assert "Notas descartadas" not in out


def test_review_lists_notes_dropped_over_invalid_citations(request_file, capsys):
    dropped = DroppedNote(
        text="El permiso puede prorrogarse.",
        section_ids=["RC-2023-E:2.2.6(c)"],
        curable=False,
        unretrieved_section_ids=["RC-2023-E:2.2.6(c)"],
    )
    uncited = DroppedNote(
        text="La solicitud está completa.",
        section_ids=[],
        curable=False,
        unretrieved_section_ids=[],
    )
    pipeline = FakePipeline(
        recommendation(notes_retried=True, dropped_notes=[dropped, uncited])
    )

    main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert (
        "Notas descartadas (citas no válidas tras un reintento):\n"
        "  ⚠ El permiso puede prorrogarse. [RC-2023-E:2.2.6(c)]: "
        "RC-2023-E:2.2.6(c) no está entre las Secciones que aplican\n"
        "  ⚠ La solicitud está completa.: no cita ninguna Sección\n"
    ) in out


def test_review_says_when_notes_were_rewritten_over_invalid_citations(request_file, capsys):
    main(["review", str(request_file)], pipeline=FakePipeline(recommendation(notes_retried=True)))

    out = capsys.readouterr().out
    assert "Notas (reescritas una vez: citas no válidas en el primer intento):\n" in out
    assert "Notas descartadas" not in out


def fee_check(raw):
    return check_fee(parse_permit_request(raw), load_fee_rules())


def test_review_prints_the_fee_check_with_each_charge_and_its_cited_item(request_file, capsys):
    check = fee_check(GOOD_PCOC | {"declared_fee": 1210})

    main(["review", str(request_file)], pipeline=FakePipeline(recommendation(fee_check=check)))

    out = capsys.readouterr().out
    assert (
        "Verificación de derechos: coinciden [COBROS:AnejoA:6]\n"
        "  Declarados: $1,210.00\n"
        "  Calculados: $1,210.00\n"
        "       $100.00  6(a) Radicación (cargo por servicio)\n"
        "       $925.00  6(b) Aranceles\n"
        "       $185.00  6(c) Sellos del CIAPR y CAAPPR\n"
        "\n"
        "Notas:"
    ) in out
    assert "Exención" not in out


def test_review_flags_a_fee_mismatch(request_file, capsys):
    check = fee_check(GOOD_PCOC)

    main(["review", str(request_file)], pipeline=FakePipeline(recommendation(fee_check=check)))

    out = capsys.readouterr().out
    assert "Verificación de derechos: ⚠ no coinciden [COBROS:AnejoA:6]\n" in out
    assert "  Declarados: $1,850.00\n  Calculados: $1,210.00\n" in out


def test_review_prints_the_exemption_applied_and_what_each_charge_was_before_it(
    request_file, capsys
):
    check = fee_check(GOOD_PCOC | {"applicant_type": "gobierno"})

    main(["review", str(request_file)], pipeline=FakePipeline(recommendation(fee_check=check)))

    out = capsys.readouterr().out
    assert "[COBROS:AnejoA:6, COBROS:TERCERO]" in out
    assert "        $25.00  6(a) Radicación (cargo por servicio) (25% de $100.00)\n" in out
    assert "       $925.00  6(b) Aranceles\n" in out
    assert "  Exención aplicada: Gobierno estatal, federal o municipal\n" in out


def test_review_says_why_a_fee_could_not_be_verified(request_file, capsys):
    check = fee_check(GOOD_PCOC | {"estimated_work_value": None})

    main(["review", str(request_file)], pipeline=FakePipeline(recommendation(fee_check=check)))

    out = capsys.readouterr().out
    assert (
        "Verificación de derechos: ⚠ no se pudo verificar: no se declaró el valor estimado "
        "de la obra [COBROS:AnejoA:6]\n"
        "  Declarados: $1,850.00\n"
        "\n"
        "Notas:"
    ) in out


def test_review_without_a_fee_check_prints_no_fee_block(request_file, capsys):
    main(["review", str(request_file)], pipeline=FakePipeline(recommendation()))

    assert "Verificación de derechos" not in capsys.readouterr().out


def test_review_prints_every_requirement_and_names_the_weakest(request_file, capsys):
    main(["review", str(request_file)], pipeline=FakePipeline(recommendation()))

    out = capsys.readouterr().out
    assert "Puntuación de aprobación: 0.86 (requisito más débil: seccion_2_1_9_2)" in out
    assert (
        "Requisitos:\n"
        "  0.91         El proponente acredita legitimación activa. [pcoc_legitimacion]\n"
        "  0.97         La obra cumple los parámetros del distrito. [pcoc_parametros]\n"
        "  0.86         Requisitos aplicables de la Sección 2.1.9.2 (REQUISITOS DE PRESENTACIÓN) "
        "[seccion_2_1_9_2]\n"
    ) in out


def test_review_marks_a_refused_requirement_as_not_assessed(request_file, capsys):
    pipeline = FakePipeline(
        recommendation(
            approval_score=None,
            verdict="manual_review",
            requirements=[criterion_requirement(PCOC_STANDING, 0.91), criterion_requirement(PCOC_PARAMETERS, None)],
        )
    )

    main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert "Puntuación de aprobación: sin puntuación\n" in out
    assert "  no evaluado  La obra cumple los parámetros del distrito. [pcoc_parametros]" in out


def test_review_prints_every_api_error_with_its_step_and_subject(request_file, capsys):
    pipeline = FakePipeline(
        recommendation(
            errors=[
                ModelFailure(
                    step="validate_sections",
                    subject="RC-2023-E:2.1.9.3",
                    message="gpt-6-luna: Connection error.",
                ),
                ModelFailure(step="write_notes", message="gpt-6-luna: Request timed out."),
            ]
        )
    )

    main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert (
        "Errores de la API (la revisión continuó sin esas respuestas):\n"
        "  ⚠ validate_sections [RC-2023-E:2.1.9.3]: gpt-6-luna: Connection error.\n"
        "  ⚠ write_notes: gpt-6-luna: Request timed out.\n"
    ) in out


def test_review_names_api_errors_as_the_reason_for_manual_review(request_file, capsys):
    failure = ModelFailure(step="write_notes", message="gpt-6-luna: Request timed out.")
    pipeline = FakePipeline(recommendation(verdict="manual_review", errors=[failure]))

    main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert "Puntuación de aprobación: 0.86" in out
    assert "Veredicto: Revisión manual (falló una llamada a la API; ver abajo)\n" in out


def test_review_without_api_errors_prints_no_errors_block(request_file, capsys):
    main(["review", str(request_file)], pipeline=FakePipeline(recommendation()))

    assert "Errores de la API" not in capsys.readouterr().out


def test_review_names_the_request_type_of_a_permiso_unico(request_file, capsys):
    pipeline = FakePipeline(
        recommendation(
            request=parse_permit_request(GOOD_PERMISO_UNICO),
            approval_score=0.95,
            requirements=[criterion_requirement(PU_USE, 0.95)],
        )
    )

    main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert "Recomendación — Solicitud de Permiso Único de Ana Lucía Torres Rivera" in out
    assert "Requisitos:\n  0.95         El uso está permitido ministerialmente. [pu_uso]\n" in out


def test_review_without_requirements_prints_no_requirements_block(request_file, capsys):
    pipeline = FakePipeline(
        recommendation(approval_score=None, verdict="manual_review", requirements=[])
    )

    main(["review", str(request_file)], pipeline=pipeline)

    assert "Requisitos:" not in capsys.readouterr().out


def test_review_counts_the_sections_that_do_not_apply_but_lists_them_only_on_request(
    request_file, capsys
):
    main(["review", str(request_file)], pipeline=FakePipeline(recommendation()))

    out = capsys.readouterr().out
    assert "Secciones recuperadas: 3 (1 no aplica; --selection para ver por qué)" in out
    assert "No se solicita una prórroga" not in out


def test_review_with_selection_lists_the_sections_that_do_not_apply(request_file, capsys):
    main(["review", "--selection", str(request_file)], pipeline=FakePipeline(recommendation()))

    out = capsys.readouterr().out
    assert (
        "Secciones que no aplican:\n"
        "  RC-2023-E:2.2.6: No se solicita una prórroga del permiso.\n"
    ) in out


def test_review_prints_how_many_clauses_were_set_aside_but_not_the_list(request_file, capsys):
    main(["review", str(request_file)], pipeline=FakePipeline(recommendation()))

    out = capsys.readouterr().out
    assert "Incisos evaluados: 1 de 3 (2 descartados; --selection para ver la lista)" in out
    assert "Solo para asuntos discrecionales" not in out


def test_review_with_selection_lists_every_clause_with_its_category_and_reason(
    request_file, capsys
):
    main(["review", "--selection", str(request_file)], pipeline=FakePipeline(recommendation()))

    out = capsys.readouterr().out
    assert "Selección de incisos:" in out
    assert (
        "  trámite del sistema  RC-2023-E:2.1.9.2(a): La presentación en el SUI la gestiona el sistema."
        in out
    )
    assert "  aplica               RC-2023-E:2.1.9.2(k): Los planos deben estar sellados" in out
    assert "  no aplica            RC-2023-E:2.1.9.2(l): Solo para asuntos discrecionales." in out


def test_review_warns_about_clauses_the_selector_missed_or_invented(request_file, capsys):
    pipeline = FakePipeline(
        recommendation(
            unclassified_clause_refs=["RC-2023-E:2.1.9.3(b)"],
            unknown_clause_refs=["RC-2023-E:2.1.9.2(z)"],
        )
    )

    main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert "⚠ Inciso sin clasificar (se evalúa igualmente): RC-2023-E:2.1.9.3(b)" in out
    assert "⚠ Inciso desconocido descartado: RC-2023-E:2.1.9.2(z)" in out


def test_review_marks_borderline_not_approve(request_file, capsys):
    pipeline = FakePipeline(recommendation(approval_score=0.45, verdict="not_approve", borderline=True))

    main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert "Puntuación de aprobación: 0.45" in out
    assert "Veredicto: No aprobar (caso límite" in out


def test_review_reports_refused_score_as_manual_review(request_file, capsys):
    pipeline = FakePipeline(recommendation(approval_score=None, verdict="manual_review"))

    main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert "Puntuación de aprobación: sin puntuación" in out
    assert "Veredicto: Sin puntuación — revisión manual" in out


def test_review_of_invalid_request_lists_errors_and_fails(request_file, capsys):
    pipeline = FakePipeline(InvalidPermitRequest(["catastro: formato no válido"]))

    exit_code = main(["review", str(request_file)], pipeline=pipeline)

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Solicitud de permiso inválida" in err
    assert "catastro: formato no válido" in err


def test_review_of_missing_file_fails_without_reviewing(tmp_path, capsys):
    pipeline = FakePipeline(recommendation())

    exit_code = main(["review", str(tmp_path / "no-existe.json")], pipeline=pipeline)

    assert exit_code == 1
    assert "no-existe.json" in capsys.readouterr().err
    assert pipeline.reviewed == []


def test_review_of_malformed_json_fails_without_reviewing(tmp_path, capsys):
    path = tmp_path / "rota.json"
    path.write_text("{no es json")
    pipeline = FakePipeline(recommendation())

    exit_code = main(["review", str(path)], pipeline=pipeline)

    assert exit_code == 1
    assert "JSON" in capsys.readouterr().err
    assert pipeline.reviewed == []


def test_review_requires_a_file_argument(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["review"])

    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["ingest", "serve"])
def test_placeholder_subcommands_report_not_implemented(command, capsys):
    exit_code = main([command])

    assert exit_code == 1
    assert f"'{command}' is not implemented yet" in capsys.readouterr().err


def test_unknown_subcommand_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_no_subcommand_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 2
