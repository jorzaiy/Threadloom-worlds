FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    THREADLOOM_HOST=0.0.0.0 \
    THREADLOOM_PORT=7860

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN grep -vE '^(pytest|basedpyright)' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY --chown=user . .
# WORKDIR is created root-owned; give the app user the tree + a writable data dir.
RUN mkdir -p /home/user/app/runtime-data && chown -R user:user /home/user/app
USER user

EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
