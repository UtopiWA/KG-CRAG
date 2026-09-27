"""跨模块共享的结构化异常。"""

from kg_crag.models import ErrorDetail


class KGCRAGError(Exception):
    """携带安全错误详情，同时避免字符串表示泄漏上下文。"""

    def __init__(self, detail: ErrorDetail) -> None:
        self.detail = detail
        super().__init__(f"{detail.code.value}: {detail.message}")
