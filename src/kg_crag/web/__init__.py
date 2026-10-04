"""可信 Web 搜索和外部 Evidence 转换。"""

from kg_crag.web.evidence import build_external_coverage, convert_web_results
from kg_crag.web.policy import TrustedSourcePolicy, build_web_query
from kg_crag.web.tavily import TavilySearchProvider

__all__ = [
    "TavilySearchProvider",
    "TrustedSourcePolicy",
    "build_external_coverage",
    "build_web_query",
    "convert_web_results",
]
