#stage 1
FROM python:3.12-slim AS builder

# Install build deps ONLY in builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# Stage 2: Final Runtime Image

FROM python:3.12-slim

WORKDIR /app

# Install wheels built in stage 1
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy  assets
COPY src/assets /app/src/
COPY data/final /app/data/
COPY dashboard.css /app/
COPY gradio_dashboard.py /app/

EXPOSE 7860

CMD ["python", "gradio_dashboard.py"]
