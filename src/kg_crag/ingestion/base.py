"""可替换文档解析器契约及确定性测试替身。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from kg_crag.models import ParsedDocument, ParseRequest


@runtime_checkable
class DocumentParser(Protocol):
    """把一个已校验的本地 PDF 转为公共结构化文档。"""

    name: str
    version: str

    def parse(self, pdf_path: Path, request: ParseRequest) -> ParsedDocument: ...


class MockDocumentParser:
    """离线测试使用的固定结果解析器。"""

    name = "mock"
    version = "1"

    def __init__(self, document: ParsedDocument) -> None:
        self._document = document
        self.calls: list[tuple[Path, ParseRequest]] = []

    def parse(self, pdf_path: Path, request: ParseRequest) -> ParsedDocument:
        self.calls.append((pdf_path, request))
        return self._document.model_copy(deep=True)
