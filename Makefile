PYTHON ?= poetry run python
OLLAMA_MODEL ?= qwen3:8b
DOCS ?= 1
CASE_CONFIG ?= config_case/case_1.yaml
MSG ?= Sync prompt templates

.PHONY: help install setup langfuse-up langfuse-down langfuse-ps ollama-health ollama-pull sync-langfuse generate generate-classic check

help:
	@echo "Common targets:"
	@echo "  make setup          Install deps, start Langfuse, pull Ollama model, sync prompts"
	@echo "  make generate       Generate documents with LangGraph workflow"
	@echo "  make langfuse-up    Start local Langfuse Docker stack"
	@echo "  make ollama-pull    Pull OLLAMA_MODEL=$(OLLAMA_MODEL)"
	@echo "  make check          Run ruff"

install:
	poetry install

setup: install langfuse-up ollama-pull sync-langfuse

langfuse-up:
	docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml up -d

langfuse-down:
	docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml down

langfuse-ps:
	docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml ps

ollama-health:
	curl -fsS http://localhost:11434/api/tags >/dev/null
	@echo "Ollama is reachable at http://localhost:11434"

ollama-pull: ollama-health
	ollama pull $(OLLAMA_MODEL)

sync-langfuse:
	$(PYTHON) -m src.synthetic_ner.sync_langfuse_prompts --commit-message "$(MSG)"

generate:
	$(PYTHON) main.py --case-config $(CASE_CONFIG) --documents $(DOCS) --workflow-mode langgraph

generate-classic:
	$(PYTHON) main.py --case-config $(CASE_CONFIG) --documents $(DOCS) --workflow-mode classic

check:
	poetry run ruff check .
