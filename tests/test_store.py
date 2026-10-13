import sqlite3

import pytest

from case_mgmt_ai_demo.determination import Determination
from case_mgmt_ai_demo.permit_request import parse_permit_request
from case_mgmt_ai_demo.store import AlreadyDetermined, PermitRequestStore
from test_cli import fee_check, recommendation
from test_review import GOOD_PCOC, GOOD_PERMISO_UNICO


def test_a_stored_permit_request_is_read_back_with_its_fields(tmp_path):
    store = PermitRequestStore(tmp_path / "app.sqlite")

    request_id = store.add(parse_permit_request(GOOD_PERMISO_UNICO))

    stored = store.get(request_id)
    assert stored is not None
    assert stored.id == request_id
    assert stored.request == parse_permit_request(GOOD_PERMISO_UNICO)


def test_permit_requests_are_listed_in_the_order_stored(tmp_path):
    store = PermitRequestStore(tmp_path / "app.sqlite")
    first = store.add(parse_permit_request(GOOD_PCOC))
    second = store.add(parse_permit_request(GOOD_PERMISO_UNICO))

    assert [stored.id for stored in store.all()] == [first, second]
    assert [stored.request.request_type for stored in store.all()] == ["PCOC", "PERMISO_UNICO"]


def test_an_unknown_permit_request_is_none(tmp_path):
    store = PermitRequestStore(tmp_path / "app.sqlite")

    assert store.get(7) is None
    assert store.recommendation(7) is None


def test_a_permit_request_has_no_recommendation_until_one_is_saved(tmp_path):
    store = PermitRequestStore(tmp_path / "app.sqlite")
    request_id = store.add(parse_permit_request(GOOD_PCOC))

    assert store.recommendation(request_id) is None


def test_permit_requests_and_recommendations_survive_a_restart(tmp_path):
    path = tmp_path / "data" / "app.sqlite"
    saved = recommendation(fee_check=fee_check(GOOD_PCOC))
    request_id = PermitRequestStore(path).add(saved.request)
    PermitRequestStore(path).save_recommendation(request_id, saved)

    reopened = PermitRequestStore(path)

    stored = reopened.get(request_id)
    assert stored is not None and stored.request == saved.request
    assert reopened.recommendation(request_id) == saved


def test_a_recommendation_this_version_cannot_read_counts_as_none(tmp_path, capsys):
    path = tmp_path / "app.sqlite"
    store = PermitRequestStore(path)
    request_id = store.add(parse_permit_request(GOOD_PCOC))
    store.save_recommendation(request_id, recommendation())
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE recommendations SET body = '{\"verdict\": \"old\"}'")

    assert store.recommendation(request_id) is None
    assert f"solicitud {request_id}" in capsys.readouterr().err


def test_saving_again_replaces_the_recommendation(tmp_path):
    store = PermitRequestStore(tmp_path / "app.sqlite")
    request_id = store.add(parse_permit_request(GOOD_PCOC))
    store.save_recommendation(request_id, recommendation())

    store.save_recommendation(request_id, recommendation(approval_score=0.12, verdict="not_approve"))

    again = store.recommendation(request_id)
    assert again is not None and again.verdict == "not_approve"


def test_a_permit_request_has_no_determination_until_one_is_saved(tmp_path):
    store = PermitRequestStore(tmp_path / "app.sqlite")
    request_id = store.add(parse_permit_request(GOOD_PCOC))

    assert store.determination(request_id) is None


def test_a_determination_survives_a_restart(tmp_path):
    path = tmp_path / "app.sqlite"
    request_id = PermitRequestStore(path).add(parse_permit_request(GOOD_PCOC))
    saved = Determination(
        kind="override", recommended="approve", outcome="not_approve", reason="Falta el plano."
    )
    PermitRequestStore(path).save_determination(request_id, saved)

    assert PermitRequestStore(path).determination(request_id) == saved


ACCEPTED = Determination(kind="accept", recommended="approve", outcome="approve")


def test_a_permit_request_has_one_determination(tmp_path):
    store = PermitRequestStore(tmp_path / "app.sqlite")
    request_id = store.add(parse_permit_request(GOOD_PCOC))
    store.save_determination(request_id, ACCEPTED)

    with pytest.raises(AlreadyDetermined):
        store.save_determination(
            request_id, ACCEPTED.model_copy(update={"kind": "cure_request", "outcome": None})
        )

    assert store.determination(request_id) == ACCEPTED


def test_a_determined_permit_request_keeps_its_recommendation(tmp_path):
    store = PermitRequestStore(tmp_path / "app.sqlite")
    request_id = store.add(parse_permit_request(GOOD_PCOC))
    store.save_recommendation(request_id, recommendation())
    store.save_determination(request_id, ACCEPTED)

    with pytest.raises(AlreadyDetermined):
        store.save_recommendation(
            request_id, recommendation(approval_score=0.12, verdict="not_approve")
        )

    kept = store.recommendation(request_id)
    assert kept is not None and kept.verdict == "approve"
