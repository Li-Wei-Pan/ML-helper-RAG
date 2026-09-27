FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv
COPY pyproject.toml .
RUN uv pip install --system .
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]