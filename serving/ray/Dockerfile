FROM rayproject/ray:2.55.1-py311-gpu

USER root

WORKDIR /serve_app

ENV UV_SYSTEM_PYTHON=1

RUN python -m pip install --no-cache-dir uv==0.7.3

COPY requirements.txt /serve_app/requirements.txt
RUN uv pip install --no-cache-dir -r /serve_app/requirements.txt

COPY entrypoint.sh /serve_app/entrypoint.sh
COPY serve_app.py /serve_app/serve_app.py

RUN chmod +x /serve_app/entrypoint.sh \
    && chown -R ray:users /serve_app

USER root

CMD ["/serve_app/entrypoint.sh"]
