import io
import json
from collections.abc import Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient

from case_mgmt_ai_demo import main, unlimited_ocr
from case_mgmt_ai_demo.cli import ProgressBar, _refs
from case_mgmt_ai_demo.fee_check import check_fee, load_fee_rules
from case_mgmt_ai_demo.permit_request import parse_permit_request
from case_mgmt_ai_demo.review import (
    ClauseSelection,
    DroppedNote,
    InvalidPermitRequest,
    MissingPinnedClause,
    Note,
    ModelFailure,
    PipelineTrace,
    Recommendation,
    RequirementAssessment,
    RetrievedSection,
    SectionRelevance,
    TraceStep,
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
        "               incisos: RC-2023-E:2.1.9.5(b)\n"
        "  0.97         La obra cumple los parámetros del distrito. [pcoc_parametros]\n"
        "               incisos: RC-2023-E:2.2.1.3(d)\n"
        "  0.86         Requisitos aplicables de la Sección 2.1.9.2 (REQUISITOS DE PRESENTACIÓN) "
        "[seccion_2_1_9_2]\n"
        "               incisos: RC-2023-E:2.1.9.2(k)\n"
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


TRACE = PipelineTrace(
    retrieved=[
        RetrievedSection(
            section_id="RC-2023-E:2.1.9.2",
            heading="REQUISITOS DE PRESENTACIÓN",
            similarity=0.5481,
            pinned_by=[],
        ),
        RetrievedSection(
            section_id="RC-2023-E:2.2.6",
            heading="PRÓRROGA A LA VIGENCIA",
            similarity=0.5312,
            pinned_by=[],
            clause_refs=["RC-2023-E:2.2.6(a)", "RC-2023-E:2.2.6(b)"],
        ),
        RetrievedSection(
            section_id="RC-2023-E:2.1.9.5",
            heading="LEGITIMACIÓN ACTIVA (STANDING)",
            similarity=None,
            pinned_by=["pcoc_legitimacion"],
        ),
    ],
    steps=[
        TraceStep(node="intake", output={"criteria": ["pcoc_legitimacion", "pcoc_parametros"]}),
        TraceStep(
            node="decide",
            output={"approval_score": 0.86, "errors": [{"step": "decide", "message": "falló"}]},
        ),
    ],
)


def test_review_without_trace_prints_no_trace(request_file, capsys):
    main(["review", str(request_file)], pipeline=FakePipeline(recommendation(trace=TRACE)))

    out = capsys.readouterr().out
    assert "Traza del pipeline" not in out
    assert "Salida de cada nodo" not in out


def test_review_with_trace_lists_each_retrieved_section_with_similarity_and_verdict(
    request_file, capsys
):
    pipeline = FakePipeline(recommendation(trace=TRACE))

    main(["review", "--trace", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert (
        "Secciones recuperadas (similitud con la solicitud):\n"
        "  0.548  RC-2023-E:2.1.9.2 REQUISITOS DE PRESENTACIÓN\n"
        "      aplica: Aplica.\n"
        "  0.531  RC-2023-E:2.2.6 PRÓRROGA A LA VIGENCIA\n"
        "      no aplica: No se solicita una prórroga del permiso.\n"
        "      incisos no evaluados: RC-2023-E:2.2.6(a), RC-2023-E:2.2.6(b)\n"
        "    —    RC-2023-E:2.1.9.5 LEGITIMACIÓN ACTIVA (STANDING) (fijada por pcoc_legitimacion)\n"
        "      aplica: Aplica.\n"
    ) in out


def test_review_with_trace_lists_every_clause_with_its_category_and_reason(request_file, capsys):
    pipeline = FakePipeline(
        recommendation(trace=TRACE, unclassified_clause_refs=["RC-2023-E:2.1.9.3(b)"])
    )

    main(["review", "--trace", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert (
        "Selección de incisos:\n"
        "  trámite del sistema  RC-2023-E:2.1.9.2(a): La presentación en el SUI la gestiona el "
        "sistema.\n"
        "  aplica               RC-2023-E:2.1.9.2(k): Los planos deben estar sellados y "
        "certificados.\n"
        "  no aplica            RC-2023-E:2.1.9.2(l): Solo para asuntos discrecionales.\n"
        "  ⚠ sin clasificar (se evalúa igualmente)  RC-2023-E:2.1.9.3(b)\n"
    ) in out
    assert out.count("Selección de incisos:") == 1


def test_review_with_trace_lists_each_requirement_with_its_clauses(request_file, capsys):
    pipeline = FakePipeline(recommendation(trace=TRACE))

    main(["review", "--trace", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert (
        "Requisitos y sus incisos:\n"
        "  0.91         pcoc_legitimacion: RC-2023-E:2.1.9.5(b)\n"
        "  0.97         pcoc_parametros: RC-2023-E:2.2.1.3(d)\n"
        "  0.86         seccion_2_1_9_2: RC-2023-E:2.1.9.2(k)\n"
    ) in out


def test_review_with_trace_prints_each_nodes_output(request_file, capsys):
    pipeline = FakePipeline(recommendation(trace=TRACE))

    main(["review", "--trace", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert out.endswith(
        "Salida de cada nodo:\n"
        "  intake\n"
        '    criteria: ["pcoc_legitimacion", "pcoc_parametros"]\n'
        "  decide\n"
        "    approval_score: 0.86\n"
        '    errors: [{"step": "decide", "message": "falló"}]\n'
    )


def test_review_with_trace_and_selection_lists_the_selection_once(request_file, capsys):
    pipeline = FakePipeline(recommendation(trace=TRACE))

    main(["review", "--trace", "--selection", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert out.count("Selección de incisos:") == 1
    assert "Secciones que no aplican:" not in out


def test_review_reports_a_criterion_pin_the_sections_do_not_hold(request_file, capsys):
    error = MissingPinnedClause("Criterion pcoc_planos pins RC-2023-E:2.1.9.2(k)")

    assert main(["review", str(request_file)], pipeline=FakePipeline(error)) == 1

    err = capsys.readouterr().err
    assert "Criterion pcoc_planos pins RC-2023-E:2.1.9.2(k)" in err
    assert "ingest" in err


def test_review_against_an_index_without_rc_2023_e_says_to_ingest(
    request_file, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PRA_RETRIEVER", "index")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert main(["review", str(request_file)]) == 1

    err = capsys.readouterr().err
    assert f"El índice del Reglamento ({tmp_path / 'chroma'}) no contiene RC-2023-E" in err
    assert "ingest" in err


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


@pytest.fixture
def served(tmp_path, monkeypatch):
    """What `serve` hands to uvicorn, with the data and the examples in a temporary directory."""
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "1-pcoc.json").write_text(json.dumps(GOOD_PCOC), encoding="utf-8")
    (examples / "2-permiso-unico.json").write_text(
        json.dumps(GOOD_PERMISO_UNICO), encoding="utf-8"
    )
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PRA_EXAMPLES_DIR", str(examples))
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **options: calls.append({"app": app} | options))
    return calls


def test_serve_starts_the_web_app_with_the_example_permit_requests(served, capsys):
    exit_code = main(["serve", "--port", "8123"], pipeline=FakePipeline(recommendation()))

    assert exit_code == 0
    [call] = served
    assert (call["host"], call["port"]) == ("127.0.0.1", 8123)
    assert "http://127.0.0.1:8123" in capsys.readouterr().err
    client = TestClient(call["app"])
    home = client.get("/").text
    assert home.index("María Isabel Rivera Ortiz") < home.index("Ana Lucía Torres Rivera")
    assert "Aún no hay una Recomendación" in client.get("/solicitudes/1").text
    assert "Aprobar" in client.post("/solicitudes/1/recomendacion").text


def test_serve_keeps_what_is_stored_when_started_again(served):
    main(["serve"], pipeline=FakePipeline(recommendation()))
    TestClient(served[0]["app"]).post("/solicitudes/2/recomendacion")

    main(["serve"], pipeline=FakePipeline(recommendation()))

    client = TestClient(served[1]["app"])
    assert client.get("/").text.count('href="/solicitudes/') == 2  # the examples, loaded once
    assert "Puntuación de aprobación" in client.get("/solicitudes/2").text


def test_serve_skips_an_example_that_is_not_a_valid_permit_request(served, tmp_path, capsys):
    (tmp_path / "examples" / "3-rota.json").write_text('{"request_type": "PCOC"}')
    (tmp_path / "examples" / "4-no-json.json").write_text("{")

    assert main(["serve"], pipeline=FakePipeline(recommendation())) == 0

    err = capsys.readouterr().err
    assert "3-rota.json" in err and "4-no-json.json" in err
    assert TestClient(served[0]["app"]).get("/").text.count('href="/solicitudes/') == 2


def test_ingest_without_index_writes_the_report_and_prints_a_summary(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path))

    exit_code = main(["ingest", "--no-index", "--no-ocr"])

    out = capsys.readouterr().out
    assert exit_code == 0
    report = json.loads((tmp_path / "RC-2023-E.ingestion_report.json").read_text())
    assert report["config"]["max_chunk_chars"] == 6000
    assert f"Fragmentos: {report['chunks']} de {report['sections']} Secciones" in out
    assert "Índice: sin tocar (--no-index)" in out
    assert "Sin OCR" in out
    assert not (tmp_path / "chroma").exists()


def test_ingest_covers_the_fee_order_as_well_as_rc_2023_e(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path))

    exit_code = main(["ingest", "--no-index", "--no-ocr"])

    out = capsys.readouterr().out
    assert exit_code == 0
    cobros = json.loads((tmp_path / "COBROS-OA-2022-07.ingestion_report.json").read_text())
    assert cobros["chunks_by_level"] == {"seccion": 10, "anejo_item": 58}
    assert out.index("Ingesta de RC-2023-E.pdf") < out.index("Ingesta de COBROS OA-2022-07.pdf")
    assert "Fragmentos: 68 de 66 Secciones (2 divididas en partes)" in out
    # The fee order draws no text as outlines, so its summary says nothing about OCR.
    assert out.count("Texto trazado como contornos") == out.count("Sin OCR") == 1


def test_ingest_of_one_document_leaves_the_other_alone(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path))

    exit_code = main(["ingest", "--no-index", "--doc", "COBROS-OA-2022-07"])

    assert exit_code == 0
    assert "Ingesta de COBROS OA-2022-07.pdf (COBROS-OA-2022-07)" in capsys.readouterr().out
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "COBROS-OA-2022-07.chunks.jsonl",
        "COBROS-OA-2022-07.ingestion_report.json",
    ]


def test_ingest_reads_outlined_text_with_the_ocr_engine_asked_for(tmp_path, monkeypatch, capsys):
    """The model is replaced by one that reads the same thing in every image."""
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(unlimited_ocr, "_load_model", lambda: lambda image: "Texto leído")

    exit_code = main(["ingest", "--no-index", "--doc", "RC-2023-E", "--ocr", "unlimited-ocr"])

    assert exit_code == 0
    report = json.loads((tmp_path / "RC-2023-E.ingestion_report.json").read_text())
    assert report["outlined_text"]["engine"] == "unlimited-ocr"
    assert "p. 494: Texto leído" in report["outlined_text"]["readings"]
    assert "Motor de OCR: unlimited-ocr" in capsys.readouterr().out


def test_the_ocr_engine_can_be_set_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PRA_OCR_ENGINE", "unlimited-ocr")
    monkeypatch.setattr(unlimited_ocr, "_load_model", lambda: lambda image: "")

    exit_code = main(["ingest", "--no-index", "--doc", "RC-2023-E"])

    assert exit_code == 0
    report = json.loads((tmp_path / "RC-2023-E.ingestion_report.json").read_text())
    assert report["outlined_text"]["engine"] == "unlimited-ocr"


def test_ingest_stops_when_the_ocr_engine_asked_for_cannot_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path))

    def no_gpu():
        raise unlimited_ocr.UnlimitedOcrUnavailable("PyTorch no encuentra una GPU")

    monkeypatch.setattr(unlimited_ocr, "_load_model", no_gpu)

    exit_code = main(["ingest", "--no-index", "--ocr", "unlimited-ocr"])

    assert exit_code == 1
    assert "unlimited-ocr: PyTorch no encuentra una GPU" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_the_progress_bar_fills_one_line_and_ends_it_when_done():
    terminal = Terminal()
    bar = ProgressBar(terminal)

    bar("extract", 1, 4)
    bar("extract", 4, 4)

    first, last = terminal.getvalue().split("\r")[1:]
    assert first.startswith("Extrayendo el texto [") and first.endswith(" 25% (1/4 páginas)")
    assert (first.count("█"), first.count("░")) == (7, 23)
    assert "░" not in last and last.endswith("100% (4/4 páginas)\n")


def test_the_progress_bar_writes_nothing_when_not_on_a_terminal():
    captured = io.StringIO()

    ProgressBar(captured)("embed", 100, 200)

    assert captured.getvalue() == ""


def test_ingest_reports_a_missing_fee_order_pdf_before_ingesting_anything(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path))

    exit_code = main(["ingest", "--no-index", "--cobros-pdf", str(tmp_path / "nope.pdf")])

    assert exit_code == 1
    assert "No se encontró el PDF de la orden de cobros" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_ingest_reports_a_missing_pdf(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path))

    exit_code = main(["ingest", "--pdf", str(tmp_path / "nope.pdf")])

    assert exit_code == 1
    assert "No se encontró el PDF del Reglamento" in capsys.readouterr().err


def test_unknown_subcommand_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_no_subcommand_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 2


def test_review_warns_of_requirements_the_form_cannot_evidence(request_file, capsys):
    unverifiable = ClauseSelection(
        ref="RC-2023-E:9.5.2.2(a)",
        category="no_verificable",
        reason="El formulario no admite un estudio de suelos.",
    )
    pipeline = FakePipeline(recommendation(selection=[unverifiable]))

    main(["review", "--selection", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert "Incisos evaluados: 0 de 1 (1 descartados" in out
    assert "⚠ 1 de los descartados son requisitos que el formulario no permite acreditar (lista al final)" in out
    assert "  no verificable       RC-2023-E:9.5.2.2(a): El formulario no admite" in out


def test_review_names_the_clauses_each_requirement_was_assessed_on(request_file, capsys):
    main(["review", str(request_file)], pipeline=FakePipeline(recommendation()))

    out = capsys.readouterr().out
    assert "[seccion_2_1_9_2]\n               incisos: RC-2023-E:2.1.9.2(k)\n" in out


def test_clauses_of_one_section_are_named_under_it_once():
    refs = ["RC-2023-E:9.5.2.2", "RC-2023-E:9.5.2.2(a)", "RC-2023-E:9.5.2.2(c)", "RC-2023-E:3.1.3(c)"]

    assert _refs(refs) == "RC-2023-E:9.5.2.2(párrafo inicial, a, c); RC-2023-E:3.1.3(c)"


def test_a_refused_requirement_shows_the_clauses_it_was_asked_about_and_why(
    request_file, capsys
):
    refused = RequirementAssessment(
        id="seccion_2_1_9_2",
        text_es="Requisitos aplicables de la Sección 2.1.9.2",
        clause_refs=["RC-2023-E:2.1.9.2(k)"],
        probability=None,
    )
    pipeline = FakePipeline(
        recommendation(requirements=[refused], approval_score=None, verdict="manual_review")
    )

    main(["review", str(request_file)], pipeline=pipeline)

    out = capsys.readouterr().out
    assert "⚠ pregunta rechazada, sin razón dada; se preguntó por:" in out
    assert "RC-2023-E:2.1.9.2(k): Los planos deben estar sellados y certificados." in out


def test_requirements_the_form_cannot_evidence_are_listed_after_the_notes(request_file, capsys):
    unverifiable = ClauseSelection(
        ref="RC-2023-E:9.5.2.2(a)",
        category="no_verificable",
        reason="El formulario no admite un estudio de suelos.",
    )

    main(["review", str(request_file)], pipeline=FakePipeline(recommendation(selection=[unverifiable])))

    out = capsys.readouterr().out
    listed = out.index("Requisitos que el formulario no permite acreditar (sin puntuar")
    assert out.index("Notas:") < listed
    assert "  RC-2023-E:9.5.2.2(a): El formulario no admite un estudio de suelos." in out[listed:]
