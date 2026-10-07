"""Split mixed question text into Markdown and block-LaTeX segments."""

from __future__ import annotations

import re
from typing import Literal, NamedTuple


class RenderSegment(NamedTuple):
    kind: Literal["markdown", "latex"]
    content: str


_MATH_DELIMITER_RE = re.compile(
    r"(\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$))",
    flags=re.DOTALL,
)

_REPEATED_ARRAY_COLUMNS_RE = re.compile(
    r"\\begin\{array\}\{\*\{\d+\}([lcr])\}(.*?)\\end\{array\}",
    flags=re.DOTALL,
)


def normalize_latex_delimiters(text: str) -> str:
    """Convert standard TeX delimiters to the dollar form used by Markdown."""
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    return re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)


def normalize_latex_expression(expression: str) -> str:
    """Convert source-LaTeX constructs unsupported by Streamlit's KaTeX."""

    def expand_array_columns(match: re.Match[str]) -> str:
        alignment, body = match.groups()
        rows = [row for row in re.split(r"\\\\", body) if row.strip()]
        column_count = max((row.count("&") + 1 for row in rows), default=1)
        columns = alignment * column_count
        return rf"\begin{{array}}{{{columns}}}{body}\end{{array}}"

    return _REPEATED_ARRAY_COLUMNS_RE.sub(expand_array_columns, expression)


def _needs_block_renderer(expression: str) -> bool:
    """Return whether an inline expression is too complex for Markdown math."""
    return "\\begin{" in expression or "\n" in expression


def split_latex_segments(text: str) -> list[RenderSegment]:
    """Preserve simple inline math and extract complex/display math blocks.

    Streamlit's Markdown renderer works well for short expressions such as
    ``$a\\in\\mathbb{R}$``. Multiline environments such as ``cases``, ``array``,
    and ``bmatrix`` are more reliable when passed directly to ``st.latex``.
    """
    if not isinstance(text, str):
        text = str(text)
    text = normalize_latex_delimiters(text)

    segments: list[RenderSegment] = []
    markdown_parts: list[str] = []

    def flush_markdown() -> None:
        combined = "".join(markdown_parts)
        if combined.strip():
            segments.append(RenderSegment("markdown", combined))
        markdown_parts.clear()

    cursor = 0
    for match in _MATH_DELIMITER_RE.finditer(text):
        markdown_parts.append(text[cursor:match.start()])
        token = match.group(0)

        if token.startswith("$$"):
            expression = normalize_latex_expression(token[2:-2].strip())
            flush_markdown()
            if expression:
                segments.append(RenderSegment("latex", expression))
        else:
            expression = normalize_latex_expression(token[1:-1].strip())
            if _needs_block_renderer(expression):
                flush_markdown()
                if expression:
                    segments.append(RenderSegment("latex", expression))
            else:
                # Whitespace immediately inside dollar delimiters is not valid
                # for Streamlit's inline-math parser. Rebuild the token from the
                # stripped expression instead of preserving the source spacing.
                markdown_parts.append(f"${expression}$")

        cursor = match.end()

    markdown_parts.append(text[cursor:])
    flush_markdown()
    return segments
