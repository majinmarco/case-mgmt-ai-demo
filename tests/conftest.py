import os
from pathlib import Path

import pytest

from case_mgmt_ai_demo.chunking import (
    Chunking,
    chunk_glossary,
    chunk_lines,
    load_ingestion_config,
)
from case_mgmt_ai_demo.extraction import Extraction, extract_lines
from case_mgmt_ai_demo.fee_order import chunk_fee_order

RC_PDF = Path(__file__).parent.parent / "docs_to_use" / "RC-2023-E.pdf"
COBROS_PDF = Path(__file__).parent.parent / "docs_to_use" / "COBROS OA-2022-07.pdf"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_model: calls a real model API; runs only with RUN_REAL_MODEL_TESTS=1"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_REAL_MODEL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="real-model test; set RUN_REAL_MODEL_TESTS=1 to run")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def rc_extraction() -> Extraction:
    """The body of the real RC-2023-E PDF, without OCR (a few seconds, once per run)."""
    config = load_ingestion_config()
    return extract_lines(RC_PDF, config.first_page, config.last_page, config.footer_top)


@pytest.fixture(scope="session")
def rc_chunking(rc_extraction) -> Chunking:
    return chunk_lines(rc_extraction.lines, load_ingestion_config())


@pytest.fixture(scope="session")
def cobros_chunking() -> Chunking:
    """The order and Anejo A of the real COBROS OA-2022-07 PDF."""
    return chunk_fee_order(COBROS_PDF, load_ingestion_config("cobros_oa_2022_07"))


@pytest.fixture(scope="session")
def glossary_chunking() -> Chunking:
    """The terms of Tomo XII of the real RC-2023-E PDF, without OCR."""
    config = load_ingestion_config()
    assert config.glossary is not None
    pages = config.glossary
    terms = extract_lines(RC_PDF, pages.first_page, pages.last_page, config.footer_top)
    return chunk_glossary(terms.lines, config)
