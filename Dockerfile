# Serving image: FastAPI + the model bundle. Training dependencies are deliberately absent.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY docker/requirements-serve.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

COPY artifacts/serving /app/bundle
RUN useradd --create-home app \
    && mkdir -p /home/app/.cache/recsys-ranker && chown -R app /home/app/.cache
USER app

ENV RECSYS_BUNDLE_DIR=/app/bundle/ml-1m \
    REDIS_URL=redis://redis:6379/0 \
    WEB_CONCURRENCY=4 \
    RANKER_BACKEND=lightgbm \
    RANKER_CACHE_DIR=/home/app/.cache/recsys-ranker \
    SERVER_SOCKETS=per-worker
EXPOSE 8000
# The compiled ranker is specialized to the CPU it runs on, so it is compiled at container
# start, once, before the workers fork; they load it from the cache. The first compile
# takes ~3 min (292 trees in one function, see recsys/serving/compiled.py); compose keeps
# the cache in a volume so restarts skip it. If compiling fails the workers serve LightGBM
# (logged; /health reports the backend in use).
# SERVER_SOCKETS: "shared" = uvicorn --workers (one listening socket, workers race to accept);
# "per-worker" = recsys.serving.launch (one SO_REUSEPORT socket per worker, kernel-balanced).
CMD ["sh", "-c", "if [ \"$RANKER_BACKEND\" = lleaves ]; then python -m recsys.serving.compiled --bundle \"$RECSYS_BUNDLE_DIR\" --cache-dir \"$RANKER_CACHE_DIR\"; fi; if [ \"$SERVER_SOCKETS\" = per-worker ]; then exec python -m recsys.serving.launch --app recsys.serving.api:app --port 8000 --workers ${WEB_CONCURRENCY}; else exec uvicorn recsys.serving.api:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --no-access-log; fi"]
