.PHONY: setup lint typecheck test test-cov ci run replay verify eval clean

VENV := .venv
PY := $(VENV)/bin/python
UV := uv

setup:
	$(UV) venv $(VENV) -p 3.11
	. $(VENV)/bin/activate && $(UV) pip install -e ".[dev]"

lint:
	. $(VENV)/bin/activate && ruff check sentinel tests

typecheck:
	. $(VENV)/bin/activate && mypy sentinel

test:
	. $(VENV)/bin/activate && pytest -q

test-cov:
	. $(VENV)/bin/activate && pytest -q --cov=sentinel --cov-report=term-missing

ci: lint typecheck test

run:
	. $(VENV)/bin/activate && sentinel run --scenario nominal_urban_loop

replay:
	. $(VENV)/bin/activate && sentinel replay $(LOG)

verify:
	. $(VENV)/bin/activate && sentinel verify $(LOG)

eval:
	. $(VENV)/bin/activate && sentinel eval --all

web:
	. $(VENV)/bin/activate && sentinel run --web --port 8765

agent:
	. $(VENV)/bin/activate && pytest -q tests/unit/test_monotonicity_fuzz.py

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache **/__pycache__ evidence_logs
