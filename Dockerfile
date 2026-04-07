FROM python:3.13-slim AS base

WORKDIR /app

# System dependencies for faiss-cpu, duckdb, and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir .

COPY . .

# Create data directories
RUN mkdir -p data/chroma data/docs data/processed

EXPOSE 8001

ENV PYTHONUNBUFFERED=1
ENV SIMPLE_MODE=true

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
