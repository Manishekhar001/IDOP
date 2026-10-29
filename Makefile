.PHONY: install test lint format run run-prod seed clean help docker-up docker-down

help:
	@echo "IDOP Development Tasks"
	@echo "======================"
	@echo "make install    - Sync locked dependencies and development tools with uv"
	@echo "make test       - Run unit tests (offline mocks)"
	@echo "make lint       - Run Ruff and Black checks"
	@echo "make format     - Format Python code with Black"
	@echo "make run        - Start the local API with hot reload"
	@echo "make run-prod   - Start the API without hot reload"
	@echo "make seed       - Initialize the database and seed benchmark docs"
	@echo "make clean      - Remove Python caches and temporary test files"
	@echo "make docker-up  - Start local PostgreSQL"

install:
	uv sync --locked --extra dev

test:
	uv run pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

lint:
	uv run ruff check app/ tests/ scripts/
	uv run black --check app/ tests/ scripts/

format:
	uv run black app/ tests/ scripts/

run:
	uv run python run.py

run-prod:
	uv run python run.py --prod

seed:
	uv run python scripts/init_db.py
	uv run python scripts/seed_benchmark_docs.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .ruff_cache

docker-up:
	docker compose up -d postgres

docker-down:
	docker compose down
