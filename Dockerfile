FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY chroma_db/ chroma_db/

ENV PATH="/app/.venv/bin:$PATH"
# Shell form: Render injects PORT (10000); fallback 8000 for local docker run.
CMD python -m uvicorn --app-dir src --host 0.0.0.0 --port ${PORT:-8000} api:app
