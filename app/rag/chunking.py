import re


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _parse_paragraphs(text: str) -> list[tuple[str, str]]:
    paragraphs: list[tuple[str, str]] = []
    headings: list[str] = []
    paragraph_lines: list[str] = []

    def add_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = " ".join(line.strip() for line in paragraph_lines).strip()
        if paragraph:
            paragraphs.append((" > ".join(headings), paragraph))
        paragraph_lines.clear()

    for line in text.splitlines():
        heading_match = HEADING_PATTERN.match(line.strip())
        if heading_match:
            add_paragraph()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            headings[level - 1 :] = [title]
        elif line.strip():
            paragraph_lines.append(line)
        else:
            add_paragraph()

    add_paragraph()
    return paragraphs


def _split_long_paragraph(
    paragraph: str, chunk_size: int, overlap: int
) -> list[str]:
    parts: list[str] = []
    start = 0

    while start < len(paragraph):
        end = min(start + chunk_size, len(paragraph))
        if end < len(paragraph):
            word_break = paragraph.rfind(" ", start, end + 1)
            if word_break > start:
                end = word_break

        part = paragraph[start:end].strip()
        if part:
            parts.append(part)
        if end == len(paragraph):
            break

        start = max(end - overlap, start + 1)
        while start < len(paragraph) and paragraph[start].isspace():
            start += 1

    return parts


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks: list[str] = []
    current_heading = ""
    current_paragraphs: list[str] = []
    current_length = 0

    def add_chunk() -> None:
        nonlocal current_length
        body = "\n\n".join(current_paragraphs).strip()
        if body:
            prefix = f"{current_heading}\n\n" if current_heading else ""
            chunks.append(f"{prefix}{body}".strip())
        current_paragraphs.clear()
        current_length = 0

    for heading, paragraph in _parse_paragraphs(text):
        if heading != current_heading:
            add_chunk()
            current_heading = heading

        if len(paragraph) > chunk_size:
            add_chunk()
            for part in _split_long_paragraph(paragraph, chunk_size, overlap):
                prefix = f"{heading}\n\n" if heading else ""
                chunks.append(f"{prefix}{part}".strip())
            continue

        separator_length = 2 if current_paragraphs else 0
        if current_length + separator_length + len(paragraph) > chunk_size:
            add_chunk()

        current_paragraphs.append(paragraph)
        current_length += separator_length + len(paragraph)

    add_chunk()
    return chunks
