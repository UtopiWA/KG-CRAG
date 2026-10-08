"""面向 API 与演示界面的薄应用服务层。"""

from kg_crag.application.live import LiveQueryApplicationService, build_live_query_service
from kg_crag.application.services import (
    ApplicationService,
    ApplicationServiceError,
    DefaultApplicationService,
    GroundedAnswerQueryAdapter,
    build_default_application_service,
)

__all__ = [
    "ApplicationService",
    "ApplicationServiceError",
    "DefaultApplicationService",
    "GroundedAnswerQueryAdapter",
    "LiveQueryApplicationService",
    "build_default_application_service",
    "build_live_query_service",
]
