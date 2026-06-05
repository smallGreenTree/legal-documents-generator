PYTHON ?= poetry run python
OLLAMA_MODEL ?= mistral-large:123b-instruct-2411-q4_K_M
DOCS ?= 1
CASE_CONFIG ?= config_case/case_1.yaml
TEMPLATE ?= templates/en_indictment.j2
MSG ?= Sync prompt templates
PREFECT_HOME ?= $(CURDIR)/.prefect
PREFECT_API_URL ?= http://localhost:4200/api
PREFECT_POOL ?= synthetic-ner-local
PREFECT_DEPLOYMENT ?= document-generation
PREFECT_QUALITY_DEPLOYMENT ?= document-quality

.PHONY: help install setup
.PHONY: langfuse-up langfuse-down langfuse-ps
.PHONY: prefect-setup prefect-up prefect-down prefect-status
.PHONY: ollama-health ollama-pull sync-langfuse
.PHONY: generate smoke-model-routes smoke-prompt-contract apple-studio-run check mi

help:
	@echo "Common targets:"
	@echo "  make setup          Install deps, prepare Prefect, start Langfuse, pull model, sync prompts"
	@echo "  make generate       Generate documents with LangGraph workflow"
	@echo "  make smoke-model-routes Check planner/writer/critic model calls"
	@echo "  make smoke-prompt-contract Check writer prompt format and content"
	@echo "  make apple-studio-run Deploy, smoke-test, then queue 10 scenario runs"
	@echo "  make mi             Show radon maintainability index for src and tests"
	@echo "  make langfuse-up    Start local Langfuse Docker stack"
	@echo "  make prefect-setup  Install/setup Prefect control plane"
	@echo "  make prefect-up     Start Prefect, deploy generation and quality flows, and run worker in background"
	@echo "  make prefect-status Show Prefect server and worker status"
	@echo "  make prefect-down   Stop Prefect worker and Docker server"
	@echo "  make ollama-pull    Pull OLLAMA_MODEL=$(OLLAMA_MODEL)"
	@echo "  make check          Run ruff"

install:
	poetry install

setup: prefect-setup langfuse-up  sync-langfuse

langfuse-up:
	docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml up -d

langfuse-down:
	docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml down

langfuse-ps:
	docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml ps

prefect-setup:
	poetry install

prefect-up:
	docker compose --env-file .env.langfuse -f docker-compose.prefect.yml up -d
	$(MAKE) _prefect-deploy
	$(MAKE) _prefect-worker-bg

prefect-down:
	@if [ -f "$(PREFECT_HOME)/run/worker.pid" ]; then \
		kill `cat $(PREFECT_HOME)/run/worker.pid` 2>/dev/null || true; \
		rm -f $(PREFECT_HOME)/run/worker.pid; \
		echo "Prefect worker stopped."; \
	else \
		echo "No Prefect worker pid file found."; \
	fi
	docker compose --env-file .env.langfuse -f docker-compose.prefect.yml down

prefect-status:
	docker compose --env-file .env.langfuse -f docker-compose.prefect.yml ps
	@if [ -f "$(PREFECT_HOME)/run/worker.pid" ]; then \
		echo "Prefect worker pid: `cat $(PREFECT_HOME)/run/worker.pid`"; \
	else \
		echo "Prefect worker: not running from pid file"; \
	fi

_prefect-deploy:
	PREFECT_HOME=$(PREFECT_HOME) PREFECT_API_URL=$(PREFECT_API_URL) \
		poetry run prefect work-pool create $(PREFECT_POOL) --type process --overwrite
	PREFECT_HOME=$(PREFECT_HOME) PREFECT_API_URL=$(PREFECT_API_URL) \
		poetry run prefect --no-prompt deploy \
		prefect_pipeline.py:generate_dataset \
		--name $(PREFECT_DEPLOYMENT) \
		--pool $(PREFECT_POOL) \
		--params '{"case_config":"$(CASE_CONFIG)","template":"$(TEMPLATE)","documents":$(DOCS),"review_scenario":true,"review_entities":true}'
	PREFECT_HOME=$(PREFECT_HOME) PREFECT_API_URL=$(PREFECT_API_URL) \
		poetry run prefect --no-prompt deploy \
		prefect_pipeline.py:score_existing_document \
		--name $(PREFECT_QUALITY_DEPLOYMENT) \
		--pool $(PREFECT_POOL) \
		--params '{"case_config":"$(CASE_CONFIG)","quality_config":"config_quality.yaml","review_document_selection":true}'

_prefect-worker-bg:
	mkdir -p $(PREFECT_HOME)/logs $(PREFECT_HOME)/run
	PREFECT_HOME=$(PREFECT_HOME) PREFECT_API_URL=$(PREFECT_API_URL) nohup poetry run prefect worker start --pool $(PREFECT_POOL) > $(PREFECT_HOME)/logs/worker.log 2>&1 & echo $$! > $(PREFECT_HOME)/run/worker.pid
	@echo "Prefect worker started in background. Log: $(PREFECT_HOME)/logs/worker.log"

ollama-health:
	curl -fsS http://localhost:11434/api/tags >/dev/null
	@echo "Ollama is reachable at http://localhost:11434"

ollama-pull: ollama-health
	ollama pull $(OLLAMA_MODEL)

sync-langfuse:
	$(PYTHON) -m src.synthetic_ner.sync_langfuse_prompts --commit-message "$(MSG)"

generate:
	$(PYTHON) main.py --case-config $(CASE_CONFIG) --template $(TEMPLATE) --documents $(DOCS) --workflow-mode langgraph

smoke-model-routes:
	$(PYTHON) scripts/smoke_model_routes.py --case-config $(CASE_CONFIG)

smoke-prompt-contract:
	$(PYTHON) scripts/smoke_prompt_contract.py --case-config $(CASE_CONFIG)

apple-studio-run:
	CASE_CONFIG=$(CASE_CONFIG) TEMPLATE=$(TEMPLATE) DOCUMENTS=10 scripts/apple_studio_prefect_run.sh

check:
	poetry run ruff check .

mi:
	poetry run radon mi src tests
