import re
from typing import Iterable, List


def _approx_token_len(text: str) -> int:
    """
    Rough token length approximation.

    This is intentionally lightweight and tokenizer-agnostic.
    It approximates tokens as:
      - word sequences
      - punctuation symbols
    """
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _split_by_delimiter(text: str, delimiter: str) -> List[str]:
    """
    Split `text` by a specific delimiter while retaining basic structure.

    - For newline-based delimiters we split directly.
    - For punctuation-based delimiters (e.g. ".", "?", "!"), we split on the
      delimiter and re-attach it to the preceding segment.
    """
    if not text:
        return []

    if delimiter in ("\n\n", "\n"):
        # Simple split that preserves non-empty segments.
        parts = [p.strip() for p in text.split(delimiter)]
        return [p for p in parts if p]

    # For punctuation / sentence-like delimiters
    pattern = f"({re.escape(delimiter)})"
    raw_parts = re.split(pattern, text)
    segments: List[str] = []
    current = ""

    for part in raw_parts:
        if part == delimiter:
            current += part
            seg = current.strip()
            if seg:
                segments.append(seg)
            current = ""
        else:
            current += part

    if current.strip():
        segments.append(current.strip())

    return [s for s in segments if s]


def _hierarchical_split(text: str, delimiters_levels: List[List[str]]) -> List[str]:
    """
    Recursively split a long text using multiple delimiter levels.

    `delimiters_levels` example:
      [
        ["\n\n"],        # big paragraphs
        [".", "?", "!"], # sentences
      ]
    """
    if not text:
        return []

    segments = [text]

    for level_delims in delimiters_levels:
        new_segments: List[str] = []
        for seg in segments:
            # If segment is already small-ish, keep it as is.
            if _approx_token_len(seg) <= 512:
                new_segments.append(seg)
                continue

            # Otherwise split by all delimiters of this level.
            to_process: List[str] = [seg]
            for d in level_delims:
                next_parts: List[str] = []
                for piece in to_process:
                    split_parts = _split_by_delimiter(piece, d)
                    if split_parts:
                        next_parts.extend(split_parts)
                    else:
                        # if splitting produced nothing, keep original piece
                        next_parts.append(piece)
                to_process = next_parts

            for p in to_process:
                p = p.strip()
                if p:
                    new_segments.append(p)

        segments = new_segments

    return [s for s in segments if s.strip()]


def recursive_split(
    text: str,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 50,
) -> List[str]:
    """
    Recursive semantic-ish text splitter.

    Goals:
    - Produce chunks around `chunk_size_tokens` tokens.
    - Use a hierarchy of delimiters:
        paragraphs -> sentences -> raw fallback.
    - Maintain an overlapping window (`chunk_overlap_tokens`) to preserve context.

    Notes:
    - Tokenization is approximated, not model-specific.
    - This function is deterministic and pure (no IO).
    """
    if not text:
        return []

    text = text.strip()
    if not text:
        return []

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")

    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative")

    approx_len = _approx_token_len(text)
    if approx_len <= chunk_size_tokens:
        return [text]

    # 1) Hierarchical pre-split into smaller semantic units.
    # These units will then be merged into final chunks of the desired size.
    delimiters_levels = [
        ["\n\n"],  # paragraphs
        [".", "?", "!"],  # sentences
    ]
    units = _hierarchical_split(text, delimiters_levels)
    if not units:
        return [text]  # fallback

    # 2) Merge units into fixed-size chunks with overlap.
    chunks: List[str] = []
    current_units: List[str] = []
    current_tokens = 0

    def flush_current():
        nonlocal current_units, current_tokens
        if not current_units:
            return
        chunk_text = " ".join(u.strip() for u in current_units if u.strip()).strip()
        if chunk_text:
            chunks.append(chunk_text)
        current_units = []
        current_tokens = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        unit_tokens = _approx_token_len(unit)

        # If a single unit is larger than chunk_size_tokens, we hard-split it.
        if unit_tokens > chunk_size_tokens * 1.5:
            # naive hard split by length
            words = re.findall(r"\w+|[^\w\s]", unit)
            start = 0
            while start < len(words):
                end = min(start + chunk_size_tokens, len(words))
                sub = " ".join(words[start:end])
                if sub.strip():
                    if current_tokens + _approx_token_len(sub) > chunk_size_tokens:
                        flush_current()
                    current_units.append(sub)
                    current_tokens += _approx_token_len(sub)
                    flush_current()
                start = end
            continue

        # Normal case: try to append to current chunk.
        if current_tokens + unit_tokens <= chunk_size_tokens:
            current_units.append(unit)
            current_tokens += unit_tokens
        else:
            # flush current chunk
            flush_current()

            # Build overlap prefix from last chunk tokens if needed.
            if chunk_overlap_tokens > 0 and chunks:
                last_chunk = chunks[-1]
                last_tokens = re.findall(r"\w+|[^\w\s]", last_chunk)
                if last_tokens:
                    overlap_slice = last_tokens[-chunk_overlap_tokens:]
                    overlap_text = " ".join(overlap_slice).strip()
                    if overlap_text:
                        overlap_tokens = _approx_token_len(overlap_text)
                        current_units = [overlap_text]
                        current_tokens = overlap_tokens
                    else:
                        current_units = []
                        current_tokens = 0
                else:
                    current_units = []
                    current_tokens = 0
            else:
                current_units = []
                current_tokens = 0

            # Now add the new unit (if it fits; otherwise it will be handled next loop)
            if unit_tokens <= chunk_size_tokens:
                current_units.append(unit)
                current_tokens += unit_tokens
            else:
                # Oversized unit: handle in next iteration path
                # by forcing split via the same logic at top.
                # To avoid infinite loops, we immediately break it up.
                words = re.findall(r"\w+|[^\w\s]", unit)
                start = 0
                while start < len(words):
                    end = min(start + chunk_size_tokens, len(words))
                    sub = " ".join(words[start:end])
                    if sub.strip():
                        if current_tokens + _approx_token_len(sub) > chunk_size_tokens:
                            flush_current()
                        current_units.append(sub)
                        current_tokens += _approx_token_len(sub)
                        flush_current()
                    start = end

    flush_current()

    # Ensure non-empty and stripped.
    return [c.strip() for c in chunks if c and c.strip()]
