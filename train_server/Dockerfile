FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /trainer

COPY train_server/backend_requirements.in /trainer/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /trainer/requirements.txt

COPY train_server train_server/
COPY packages packages/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn train_server.app:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000} --log-level ${LOG_LEVEL:-info}"]
