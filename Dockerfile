# Multi-stage build for smaller image
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev, frozen)
RUN uv sync --frozen --no-dev


# Production image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Set Python path for module imports
ENV PYTHONPATH="/app/src"

# Run in production mode
# PORT and DATABASE_URL provided by Railway
CMD ["python", "-m", "app.main", "--mode", "prod", "--host", "0.0.0.0"]
