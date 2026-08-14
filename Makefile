.PHONY: install install-dev test lint typecheck doctor docker-build docker-doctor docker-test clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev,data]"

test:
	python -m pytest

lint:
	python -m ruff check .

typecheck:
	python -m mypy src

doctor:
	python -m slytrade.cli doctor

docker-build:
	docker compose build slytrade

docker-doctor:
	docker compose run --rm slytrade doctor

docker-test:
	docker compose run --rm --build dev pytest

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
