from app.rag.chunking import chunk_text


SAMPLE_POLICY = """# Returns Policy

## Return window

Items may be returned within 30 days of delivery.

The item must be unused and in its original packaging.

## Final sale

Final-sale items cannot be returned.
"""


def test_chunk_text_is_deterministic() -> None:
    first_result = chunk_text(SAMPLE_POLICY, chunk_size=80, overlap=10)
    second_result = chunk_text(SAMPLE_POLICY, chunk_size=80, overlap=10)

    assert first_result == second_result


def test_chunk_text_never_returns_empty_chunks() -> None:
    chunks = chunk_text(SAMPLE_POLICY, chunk_size=40, overlap=5)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert chunk_text("\n\n# Empty document\n\n", 40, 5) == []


def test_chunk_text_respects_chunk_size_excluding_heading() -> None:
    long_paragraph = " ".join(["returnable"] * 30)
    text = f"# Policy\n\n## Details\n\n{long_paragraph}"
    chunks = chunk_text(text, chunk_size=50, overlap=10)

    for chunk in chunks:
        _, separator, body = chunk.partition("\n\n")
        assert separator
        assert len(body) <= 50


def test_each_chunk_carries_its_heading() -> None:
    chunks = chunk_text(SAMPLE_POLICY, chunk_size=80, overlap=10)

    assert chunks[0].startswith("Returns Policy > Return window\n\n")
    assert chunks[1].startswith("Returns Policy > Return window\n\n")
    assert chunks[2].startswith("Returns Policy > Final sale\n\n")
