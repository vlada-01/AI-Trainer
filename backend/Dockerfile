FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend_requirements.in /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app_src.app:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000} --log-level info"]

# CMD ["python", "-m", "uvicorn", "app_src.app:app", "--host=0.0.0.0", "--port=8000", "--log-level=info"]