"""科研文本的确定性 Sparse 规范化与安全查询构造。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from kg_crag.errors import KGCRAGError
from kg_crag.models import ErrorCode, ErrorDetail

TOKENIZER_VERSION = "scientific-unicode-v1"
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class PreparedSparseQuery:
    """只包含分词器生成 Token 的安全 FTS 查询。"""

    normalized: str
    tokens: tuple[str, ...]
    fts_query: str


def normalize_scientific_text(text: str) -> str:
    """以 NFC 和 case-fold 消除平台及大小写差异。"""

    return unicodedata.normalize("NFC", text).casefold().strip()


def tokenize_scientific_text(text: str) -> tuple[str, ...]:
    """保留科研标识符内部连字符和下划线，其余符号作为边界。"""

    return tuple(_TOKEN_PATTERN.findall(normalize_scientific_text(text)))


def prepare_sparse_query(
    query: str,
    *,
    max_chars: int,
    max_tokens: int,
) -> PreparedSparseQuery:
    """校验硬上限，并把 Token 逐个转义为 OR 表达式。"""

    if max_chars <= 0 or max_tokens <= 0:
        raise _query_error("query limits must be positive")
    normalized = normalize_scientific_text(query)
    if not normalized:
        raise _query_error("sparse query must not be empty")
    if len(normalized) > max_chars:
        raise _query_error(
            "sparse query exceeds character limit",
            {"actual_chars": len(normalized), "max_chars": max_chars},
        )
    tokens = tokenize_scientific_text(normalized)
    if not tokens:
        raise _query_error("sparse query contains no searchable tokens")
    if len(tokens) > max_tokens:
        raise _query_error(
            "sparse query exceeds token limit",
            {"actual_count": len(tokens), "max_count": max_tokens},
        )
    # 双引号转义即使当前 tokenizer 不产生引号也保留，防止未来规则扩展后注入 FTS 语法。
    escaped = tuple(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
    return PreparedSparseQuery(normalized, tokens, " OR ".join(escaped))


def _query_error(
    message: str,
    context: dict[str, str | int | float | bool | None] | None = None,
) -> KGCRAGError:
    return KGCRAGError(
        ErrorDetail(
            code=ErrorCode.VALIDATION,
            message=message,
            retryable=False,
            context=context or {},
        )
    )
