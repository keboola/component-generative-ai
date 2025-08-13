FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PYTHONIOENCODING=utf-8

RUN apt-get update && apt-get install -y git

WORKDIR /code/

COPY pyproject.toml .
COPY uv.lock .

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
RUN uv sync --all-groups --frozen
RUN uv add flake8

COPY /src /code/src/
COPY /tests /code/tests/
COPY /scripts /code/scripts/
COPY flake8.cfg /code/flake8.cfg
COPY deploy.sh /code/deploy.sh

CMD ["python", "-u", "/code/src/component.py"]
