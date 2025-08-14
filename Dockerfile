FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y git

WORKDIR /code/

COPY pyproject.toml .

RUN uv pip install --system .

COPY src/ .
COPY tests/ .
COPY scripts/ .
COPY flake8.cfg .
COPY deploy.sh .

CMD ["python", "-u", "/code/src/component.py"]
