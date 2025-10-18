init:
	uv sync --all-extras --dev
	uv run pre-commit install

test:
	uv run coverage run -m pytest

livedocs:
	sphinx-autobuild docs docs/_build --watch src/critical_es_value
