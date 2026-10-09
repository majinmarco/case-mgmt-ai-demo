import json
from collections.abc import Mapping
from typing import Any

import pytest

from case_mgmt_ai_demo import main
from case_mgmt_ai_demo.permit_request import PermitRequest
from case_mgmt_ai_demo.review import ClauseSelection, InvalidPermitRequest, Note, Recommendation
from test_review import GOOD_PCOC


class FakePipeline:
    def __init__(self, outcome: Recommendation | Exception):
        self.outcome = outcome
        self.reviewed: list[Mapping[str, Any]] = []

    def review(self, raw: Mapping[str, Any]) -> Recommendation:
        self.reviewed.append(raw)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def recommendation(**overrides) -> Recommendation:
    fields = dict(
        request=PermitRequest.model_validate(GOOD_PCOC),
        approval_score=0.86,
        verdict="approve",
        borderline=False,
        notes=[
            Note(
                text="La solicitante es dueña y presenta copia de la escritura pública.",
                section_ids=["RC-2023-E:2.1.9.5(b)"],
                curable=False,
            ),
            Note(
                text="Falta la determinación de cumplimiento ambiental.",
                section_ids=["RC-2023-E:2.1.9.2(g)", "RC-2023-E:3.1.3.2"],
                curable=True,
                unretrieved_section_ids=["RC-2023-E:3.1.3.2"],
            ),
        ],
        retrieved_section_ids=["RC-2023-E:2.1.9.2", "RC-2023-E:2.1.9.5"],
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
    assert "[RC-2023-E:2.1.9.2(g), RC-2023-E:3.1.3.2] (subsanable)" in out
    assert "Cita no verificada: RC-2023-E:3.1.3.2" in out


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
