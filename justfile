set shell := ["bash", "-cu"]

venv:
    uv venv

sync:
    uv sync

test:
    uv run pytest -q tests.py
