.PHONY: install test lint run seed clean help

help:
	@echo "IDOP Development Tasks"
	@echo "======================"
	@echo "make install    - Install dependencies from requirements.txt"
	@echo "make test       - Run unit tests (offline mocks)"
	@echo "make lint       - Run linters (ruff check + black check)"
	@echo "make format     - Auto-format code with black"
	@echo "make run        - Start dev server with hot-reload"
	@echo "make run-prod   - Start production server (no hot-reload)"
	@echo "make seed       - Initialize database and seed benchmark docs"
	@echo "make clean      - Remove __pycache__, .pyc, and temp files"
	@echo "make docker-up  - Start infrastructure (Postgres + dependencies)"

install:
	pip install -r requirements.txt
	pip install litellm langchain-litellm langchain-voyageai langchain-nomic langchain-groq

test:
	pytest -xvs

lint:
	ruff check app/ tests/ scripts/
	black --check app/ tests/ scripts/

format:
	black app/ tests/ scripts/

run:
	python run.py

run-prod:
	python run.py --prod

seed:
	python scripts/init_db.py
	python scripts/seed_benchmark_docs.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .ruff_cache

docker-up:
	docker compose up -d postgres

docker-down:
	docker compose down
