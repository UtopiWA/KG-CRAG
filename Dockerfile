FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install ".[demo]"

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000 7860
CMD ["kg-crag-api"]
