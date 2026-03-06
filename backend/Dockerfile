# syntax=docker/dockerfile:1

FROM vllm/vllm-openai:latest AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    MODELS_DIR=/workspace/models
WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY front/public/. ./app/static/front/
COPY runpod-start.sh /app/runpod-start.sh
RUN chmod +x /app/runpod-start.sh

EXPOSE 8080
ENTRYPOINT ["/app/runpod-start.sh"]
