"""FastAPI 应用入口。"""

from fastapi import FastAPI

from kg_crag import __version__
from kg_crag.settings import get_settings

app = FastAPI(title="KG-CRAG API", version=__version__)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """提供不依赖外部服务的进程健康检查。"""

    settings = get_settings()
    return {"status": "ok", "version": __version__, "environment": settings.env}


def run() -> None:
    """通过安装后的命令行入口启动开发服务器。"""

    import uvicorn

    settings = get_settings()
    uvicorn.run("kg_crag.api.app:app", host=settings.api_host, port=settings.api_port)
