PYTHON ?= poetry run python
PREFECT ?= poetry run prefect
DOCS ?= 1
CASE_CONFIG ?= config_case/case_1.yaml
TEMPLATE ?= templates/en_indictment.j2
MSG ?= Sync prompt templates
GROUNDTRUTH_DIRECTORY ?= output
GROUNDTRUTH_CONTRACT ?= groundtruth_contract.yaml
PREFECT_HOME ?= $(CURDIR)/.prefect
PREFECT_API_URL ?= http://localhost:4200/api
PREFECT_POOL ?= synthetic-ner-local
PREFECT_DEPLOYMENT ?= document-generation
PREFECT_GROUNDTRUTH_DEPLOYMENT ?= generate-groundtruth
PREFECT_MORPHOLOGY_DEPLOYMENT ?= morphological-augmentation
PREFECT_WORKER_NAME ?= synthetic-ner-worker
MORPHOLOGY_INPUT ?= output
COVERAGE_MIN ?= 59
COMPLEXITY_MAX ?= 23
QUALITY_PATHS ?= src tests main.py prefect_pipeline.py scripts/smoke_model_routes.py scripts/smoke_prompt_contract.py scripts/check_complexity.py

.PHONY: help install generate groundtruth morphology sync-mlflow generator-deploy generator-worker
.PHONY: smoke-model-routes smoke-prompt-contract
.PHONY: format format-check test coverage lint complexity sast dependency-audit security
.PHONY: check ci-quality pre-commit-install pre-commit docker-build

help:
	@echo "Common targets:"
	@echo "  make install          Install application dependencies"
	@echo "  make generate         Generate documents with the LangGraph workflow"
	@echo "  make groundtruth      Generate ground truth for GROUNDTRUTH_DIRECTORY=$(GROUNDTRUTH_DIRECTORY)"
	@echo "  make morphology       Open the Prefect morphology selection dialogue"
	@echo "  make sync-mlflow      Synchronize prompt templates to MLflow"
	@echo "  make generator-deploy Register the application flows with Prefect"
	@echo "  make generator-worker Start the application Prefect worker"
	@echo "  make check            Run formatting, lint, and tests"
	@echo "  make ci-quality       Run deterministic CI quality gates"
	@echo "  make security         Run Python SAST and dependency audit"
	@echo "  make docker-build     Build the non-root generator image"

install:
	poetry install

generator-deploy:
	@test -f "$(CASE_CONFIG)" || (echo "Missing CASE_CONFIG file: $(CASE_CONFIG)" && exit 1)
	@test -f "$(TEMPLATE)" || (echo "Missing TEMPLATE file: $(TEMPLATE)" && exit 1)
	@test -f "$(GROUNDTRUTH_CONTRACT)" || (echo "Missing GROUNDTRUTH_CONTRACT file: $(GROUNDTRUTH_CONTRACT)" && exit 1)
	mkdir -p "$(PREFECT_HOME)"
	PREFECT_HOME="$(PREFECT_HOME)" PREFECT_API_URL="$(PREFECT_API_URL)" \
		$(PREFECT) work-pool create "$(PREFECT_POOL)" --type process --overwrite
	PREFECT_HOME="$(PREFECT_HOME)" PREFECT_API_URL="$(PREFECT_API_URL)" \
		$(PREFECT) --no-prompt deploy \
		prefect_pipeline.py:generate_dataset \
		--name "$(PREFECT_DEPLOYMENT)" \
		--pool "$(PREFECT_POOL)" \
		--params '{"case_config":"$(CASE_CONFIG)","template":"$(TEMPLATE)","documents":$(DOCS),"review_scenario":true,"review_entities":true}'
	PREFECT_HOME="$(PREFECT_HOME)" PREFECT_API_URL="$(PREFECT_API_URL)" \
		$(PREFECT) --no-prompt deploy \
		prefect_pipeline.py:generate_groundtruth_directory \
		--name "$(PREFECT_GROUNDTRUTH_DEPLOYMENT)" \
		--pool "$(PREFECT_POOL)" \
		--params '{"input_directory":"$(GROUNDTRUTH_DIRECTORY)","contract_path":"$(GROUNDTRUTH_CONTRACT)"}'
	PREFECT_HOME="$(PREFECT_HOME)" PREFECT_API_URL="$(PREFECT_API_URL)" \
		$(PREFECT) --no-prompt deploy \
		prefect_pipeline.py:generate_morphological_variations \
		--name "$(PREFECT_MORPHOLOGY_DEPLOYMENT)" \
		--pool "$(PREFECT_POOL)" \
		--params '{"input_path":"","review":true,"active_to_passive":true,"verbal_to_nominal":true,"possessive_reframe":true,"intentional_typos":false,"random_layout":false,"style":"","style_temperature":0.8,"reformat_with_style":true}'

generator-worker:
	mkdir -p "$(PREFECT_HOME)"
	PREFECT_HOME="$(PREFECT_HOME)" PREFECT_API_URL="$(PREFECT_API_URL)" \
		$(PREFECT) worker start --pool "$(PREFECT_POOL)" --name "$(PREFECT_WORKER_NAME)"

sync-mlflow:
	$(PYTHON) -m src.synthetic_ner.integrations.mlflow_prompts --commit-message "$(MSG)"

generate:
	$(PYTHON) main.py --case-config "$(CASE_CONFIG)" --template "$(TEMPLATE)" --documents "$(DOCS)" --workflow-mode langgraph

groundtruth:
	$(PYTHON) prefect_pipeline.py \
		--groundtruth-directory "$(GROUNDTRUTH_DIRECTORY)" \
		--groundtruth-contract "$(GROUNDTRUTH_CONTRACT)"

morphology:
	$(PYTHON) prefect_pipeline.py \
		--morphology \
		--morphology-input "$(MORPHOLOGY_INPUT)" \
		--review-morphology

smoke-model-routes:
	$(PYTHON) scripts/smoke_model_routes.py --case-config "$(CASE_CONFIG)"

smoke-prompt-contract:
	$(PYTHON) scripts/smoke_prompt_contract.py --case-config "$(CASE_CONFIG)"

format:
	poetry run ruff format $(QUALITY_PATHS)

format-check:
	poetry run ruff format --check $(QUALITY_PATHS)

test:
	poetry run pytest -q

coverage:
	poetry run pytest -q --cov=src.synthetic_ner --cov-branch \
		--cov-report=term-missing --cov-report=xml --cov-fail-under=$(COVERAGE_MIN)

lint:
	poetry run ruff check $(QUALITY_PATHS)

complexity:
	$(PYTHON) scripts/check_complexity.py --max $(COMPLEXITY_MAX) src
	poetry run radon cc src -s -a
	poetry run radon mi src -s

sast:
	poetry run bandit -q -r src main.py prefect_pipeline.py -lll -iii

dependency-audit:
	poetry run pip-audit

security: sast dependency-audit

check: format-check lint test

ci-quality: format-check lint coverage complexity

pre-commit-install:
	poetry run pre-commit install --install-hooks
	poetry run pre-commit install --hook-type pre-push

pre-commit:
	poetry run pre-commit run --all-files

docker-build:
	docker build --tag synthetic-ner:local .
