PYTHON ?= poetry run python
OLLAMA_MODEL ?= qwen3:8b
DOCS ?= 1
CASE_CONFIG ?= config_case/case_1.yaml
TEMPLATE ?= templates/en_indictment.j2
MSG ?= Sync prompt templates
PREFECT_HOME ?= $(CURDIR)/.prefect
PREFECT_API_URL ?= http://localhost:4200/api
PREFECT_POOL ?= synthetic-ner-local
PREFECT_DEPLOYMENT ?= document-generation
PREFECT_QUALITY_DEPLOYMENT ?= document-quality
PLATFORM_ENV ?= .env.platform
PLATFORM_COMPOSE ?= docker compose --env-file $(PLATFORM_ENV) -f docker-compose.platform.yml

.PHONY: help install setup
.PHONY: platform-up platform-down platform-status platform-db-init platform-health platform-check-config
.PHONY: mlflow-up mlflow-down mlflow-status
.PHONY: prefect-setup prefect-up prefect-down prefect-status
.PHONY: ollama-health ollama-pull sync-mlflow
.PHONY: generate smoke-model-routes smoke-prompt-contract apple-studio-run check mi

help:
	@echo "Common targets:"
	@echo "  make setup          Install deps, start Prefect/MLflow/PostgreSQL, pull model, sync prompts"
	@echo "  make generate       Generate documents with LangGraph workflow"
	@echo "  make smoke-model-routes Check planner/writer/critic model calls"
	@echo "  make smoke-prompt-contract Check writer prompt format and content"
	@echo "  make apple-studio-run Deploy, smoke-test, then queue 10 scenario runs"
	@echo "  make mi             Show radon maintainability index for src and tests"
	@echo "  make platform-up    Start PostgreSQL, Prefect server, and MLflow server"
	@echo "  make platform-health Check Prefect and MLflow HTTP health endpoints"
	@echo "  make mlflow-up      Start PostgreSQL and MLflow server"
	@echo "  make prefect-setup  Install/setup Prefect control plane"
	@echo "  make prefect-up     Start Prefect, deploy generation, quality, and worker"
	@echo "  make prefect-status Show Prefect server and worker status"
	@echo "  make prefect-down   Stop Prefect worker and Docker server"
	@echo "  make ollama-pull    Pull OLLAMA_MODEL=$(OLLAMA_MODEL)"
	@echo "  make check          Run ruff"

install:
	poetry install

setup: prefect-setup platform-up ollama-pull sync-mlflow

platform-check-config:
	@test -f "$(PLATFORM_ENV)" || { echo "Missing $(PLATFORM_ENV); copy .env.platform.example and set passwords."; exit 1; }
	@if grep -Eq '^[A-Z0-9_]*PASSWORD=replace-me$$' "$(PLATFORM_ENV)"; then \
		echo "Refusing to start: replace all placeholder passwords in $(PLATFORM_ENV)."; \
		exit 1; \
	fi

platform-up: platform-check-config
	$(PLATFORM_COMPOSE) up -d postgres
	$(MAKE) platform-db-init
	$(PLATFORM_COMPOSE) up -d prefect-server mlflow-server

platform-db-init: platform-check-config
	$(PLATFORM_COMPOSE) exec -T postgres bash /docker-entrypoint-initdb.d/10-platform-databases.sh

platform-health:
	curl -fsS http://localhost:4200/api/health >/dev/null
	@echo "Prefect API is healthy: http://localhost:4200/api"
	curl -fsS http://localhost:5000/health >/dev/null
	@echo "MLflow is healthy: http://localhost:5000"

platform-down:
	$(PLATFORM_COMPOSE) down

platform-status:
	$(PLATFORM_COMPOSE) ps

mlflow-up: platform-check-config
	$(PLATFORM_COMPOSE) up -d postgres
	$(MAKE) platform-db-init
	$(PLATFORM_COMPOSE) up -d mlflow-server

mlflow-down:
	$(PLATFORM_COMPOSE) stop mlflow-server

mlflow-status:
	$(PLATFORM_COMPOSE) ps postgres mlflow-server

prefect-setup:
	poetry install

prefect-up:
	$(PLATFORM_COMPOSE) up -d postgres prefect-server mlflow-server
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
	$(PLATFORM_COMPOSE) down

prefect-status:
	$(PLATFORM_COMPOSE) ps
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

sync-mlflow:
	$(PYTHON) -m src.synthetic_ner.sync_mlflow_prompts --commit-message "$(MSG)"

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
