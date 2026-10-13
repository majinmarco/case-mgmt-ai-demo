import hashlib
from collections.abc import Sequence

import pytest

from case_mgmt_ai_demo.chunks import Chunk, load_fixture_chunks
from case_mgmt_ai_demo.index import ReglamentoIndex, SyncResult


class FakeEmbedder:
    """Deterministic vectors, and a record of every text it was asked to embed."""

    def __init__(self, model: str = "fake-embedding"):
        self.model = model
        self.embedded: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded += texts
        return [[byte / 255 for byte in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


@pytest.fixture
def chunks() -> list[Chunk]:
    return load_fixture_chunks()


def test_sync_stores_every_chunk_and_a_section_id_gives_back_its_chunk(tmp_path, chunks):
    index = ReglamentoIndex(tmp_path / "chroma")

    result = index.sync("RC-2023-E", chunks, FakeEmbedder())

    assert result == SyncResult(embedded=len(chunks), unchanged=0, removed=0)
    assert index.count() == len(chunks)
    [standing] = [c for c in chunks if c.section_id == "RC-2023-E:2.1.9.5"]
    assert index.section("RC-2023-E:2.1.9.5") == [standing]


def test_a_section_id_resolves_to_its_document_and_pages(tmp_path, chunks):
    index = ReglamentoIndex(tmp_path / "chroma")
    index.sync("RC-2023-E", chunks, FakeEmbedder())

    [general] = index.section("RC-2023-E:3.7.1.1")

    assert (general.doc, general.page_start, general.page_end) == ("RC-2023-E", 247, 248)


def test_the_parts_of_a_split_section_come_back_in_order(tmp_path, chunks):
    index = ReglamentoIndex(tmp_path / "chroma")
    index.sync("RC-2023-E", list(reversed(chunks)), FakeEmbedder())

    parts = index.section("RC-2023-E:2.1.9.2")

    assert [(part.part, part.page_start) for part in parts] == [("1-of-2", 77), ("2-of-2", 79)]


def test_an_unknown_section_id_gives_nothing(tmp_path, chunks):
    index = ReglamentoIndex(tmp_path / "chroma")
    index.sync("RC-2023-E", chunks, FakeEmbedder())

    assert index.section("RC-2023-E:9.9.9.9") == []


def test_the_heading_and_parents_are_embedded_with_the_text(tmp_path, chunks):
    embedder = FakeEmbedder()

    ReglamentoIndex(tmp_path / "chroma").sync("RC-2023-E", chunks[2:3], embedder)

    assert embedder.embedded == [
        "Tomo II > Capítulo 2.1 > Regla 2.1.9\nVALOR ESTIMADO DE LA OBRA\n" + chunks[2].text
    ]


def test_syncing_the_same_chunks_again_embeds_nothing_and_changes_nothing(tmp_path, chunks):
    ReglamentoIndex(tmp_path / "chroma").sync("RC-2023-E", chunks, FakeEmbedder())
    again = FakeEmbedder()
    reopened = ReglamentoIndex(tmp_path / "chroma")

    result = reopened.sync("RC-2023-E", chunks, again)

    assert result == SyncResult(embedded=0, unchanged=len(chunks), removed=0)
    assert again.embedded == []
    assert reopened.count() == len(chunks)


def test_only_a_changed_chunk_is_embedded_again_and_a_dropped_one_is_removed(tmp_path, chunks):
    index = ReglamentoIndex(tmp_path / "chroma")
    index.sync("RC-2023-E", chunks, FakeEmbedder())
    corrected = chunks[2].model_copy(update={"text": chunks[2].text + " Texto corregido."})
    again = FakeEmbedder()

    result = index.sync("RC-2023-E", [*chunks[:2], corrected, *chunks[3:-1]], again)

    assert result == SyncResult(embedded=1, unchanged=len(chunks) - 2, removed=1)
    assert len(again.embedded) == 1
    assert index.section(corrected.section_id) == [corrected]
    assert index.section(chunks[-1].section_id) == []


def test_another_embedding_model_embeds_everything_again(tmp_path, chunks):
    index = ReglamentoIndex(tmp_path / "chroma")
    index.sync("RC-2023-E", chunks, FakeEmbedder("model-a"))

    result = index.sync("RC-2023-E", chunks, FakeEmbedder("model-b"))

    assert result == SyncResult(embedded=len(chunks), unchanged=0, removed=0)


def test_syncing_one_document_leaves_the_chunks_of_another_alone(tmp_path, chunks):
    index = ReglamentoIndex(tmp_path / "chroma")
    fee_item = chunks[0].model_copy(
        update={"section_id": "COBROS:AnejoA:6", "doc": "COBROS-OA-2022-07", "part": None}
    )
    index.sync("COBROS-OA-2022-07", [fee_item], FakeEmbedder())

    result = index.sync("RC-2023-E", chunks[:3], FakeEmbedder())

    assert result.removed == 0
    assert index.section("COBROS:AnejoA:6") == [fee_item]


def test_two_chunks_with_the_same_section_id_and_part_are_refused(tmp_path, chunks):
    index = ReglamentoIndex(tmp_path / "chroma")

    with pytest.raises(ValueError, match="share a Section ID"):
        index.sync("RC-2023-E", [chunks[2], chunks[2]], FakeEmbedder())


class AxisEmbedder:
    """Embeds a text on the axis of the first keyword it holds (whatever the case): texts
    sharing a keyword point the same way, and any other text is unrelated to them."""

    model = "axis-embedding"

    def __init__(self, *keywords: str):
        self.keywords = [keyword.lower() for keyword in keywords]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * (len(self.keywords) + 1)
        for position, keyword in enumerate(self.keywords):
            if keyword in text.lower():
                vector[position] = 1.0
                return vector
        vector[-1] = 1.0  # none of the keywords: its own direction
        return vector


def test_search_gives_the_nearest_sections_first_with_their_similarity(tmp_path, chunks):
    embedder = AxisEmbedder("VALOR ESTIMADO DE LA OBRA")
    index = ReglamentoIndex(tmp_path / "chroma")
    index.sync("RC-2023-E", chunks, embedder)

    hits = index.search(embedder.vector("VALOR ESTIMADO DE LA OBRA"), "PCOC", top_k=3)

    assert len(hits) == 3
    assert hits[0].chunk.section_id == "RC-2023-E:2.1.9.3"
    assert hits[0].similarity == pytest.approx(1.0)
    assert hits[1].similarity == pytest.approx(0.0)


def test_search_leaves_out_sections_of_another_request_type(tmp_path, chunks):
    embedder = AxisEmbedder("VALOR ESTIMADO DE LA OBRA")
    index = ReglamentoIndex(tmp_path / "chroma")
    index.sync("RC-2023-E", chunks, embedder)
    query = embedder.vector("VALOR ESTIMADO DE LA OBRA")  # Sección 2.1.9.3 is PCOC only

    hits = index.search(query, "PERMISO_UNICO", top_k=len(chunks))

    found = {hit.chunk.section_id for hit in hits}
    assert "RC-2023-E:2.1.9.3" not in found
    assert found == {c.section_id for c in chunks if "PERMISO_UNICO" in c.request_types}


def test_search_covers_the_body_of_rc_2023_e_not_the_glossary_or_the_fee_order(tmp_path, chunks):
    embedder = AxisEmbedder("VALOR ESTIMADO DE LA OBRA")
    index = ReglamentoIndex(tmp_path / "chroma")
    value = next(c for c in chunks if c.section_id == "RC-2023-E:2.1.9.3")
    term = value.model_copy(
        update={"section_id": "RC-2023-E:Glosario:V.1", "level": "glossary_term"}
    )
    fee_item = value.model_copy(
        update={"section_id": "COBROS:AnejoA:6", "doc": "COBROS-OA-2022-07", "level": "anejo_item"}
    )
    index.sync("RC-2023-E", [value, term], embedder)
    index.sync("COBROS-OA-2022-07", [fee_item], embedder)

    hits = index.search(embedder.vector("VALOR ESTIMADO DE LA OBRA"), "PCOC", top_k=5)

    assert [hit.chunk.section_id for hit in hits] == ["RC-2023-E:2.1.9.3"]


def test_sections_gives_every_part_of_each_section_asked_for_in_that_order(tmp_path, chunks):
    index = ReglamentoIndex(tmp_path / "chroma")
    index.sync("RC-2023-E", list(reversed(chunks)), FakeEmbedder())

    found = index.sections(["RC-2023-E:3.7.1.1", "RC-2023-E:2.1.9.2", "RC-2023-E:9.9.9.9"])

    assert [(c.section_id, c.part) for c in found] == [
        ("RC-2023-E:3.7.1.1", None),
        ("RC-2023-E:2.1.9.2", "1-of-2"),
        ("RC-2023-E:2.1.9.2", "2-of-2"),
    ]
    assert index.sections([]) == []
