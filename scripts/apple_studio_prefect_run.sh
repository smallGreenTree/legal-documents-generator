#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CASE_CONFIG="${CASE_CONFIG:-config_case/case_1.yaml}"
TEMPLATE="${TEMPLATE:-templates/en_indictment.j2}"
DOCUMENTS="${DOCUMENTS:-10}"
SMOKE_DOCUMENTS="${SMOKE_DOCUMENTS:-1}"
PREFECT_HOME="${PREFECT_HOME:-$ROOT_DIR/.prefect}"
PREFECT_API_URL="${PREFECT_API_URL:-http://localhost:4200/api}"
PREFECT_POOL="${PREFECT_POOL:-synthetic-ner-local}"
PREFECT_DEPLOYMENT="${PREFECT_DEPLOYMENT:-document-generation}"
RUN_NAME="${RUN_NAME:-apple-studio-10-documents}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-apple-studio-smoke-document}"
WATCH_SMOKE="${WATCH_SMOKE:-true}"
SMOKE_WATCH_INTERVAL="${SMOKE_WATCH_INTERVAL:-15}"
SMOKE_WATCH_TIMEOUT="${SMOKE_WATCH_TIMEOUT:-7200}"
START_IN="${START_IN:-}"

export PREFECT_HOME PREFECT_API_URL

echo "== Apple Studio synthetic NER run =="
echo "case_config=$CASE_CONFIG"
echo "template=$TEMPLATE"
echo "documents=$DOCUMENTS"
echo "prefect_api=$PREFECT_API_URL"
echo

echo "== 1/4 Deploy Prefect generation flow =="
make _prefect-deploy \
  CASE_CONFIG="$CASE_CONFIG" \
  TEMPLATE="$TEMPLATE" \
  DOCS="$DOCUMENTS" \
  PREFECT_HOME="$PREFECT_HOME" \
  PREFECT_API_URL="$PREFECT_API_URL" \
  PREFECT_POOL="$PREFECT_POOL" \
  PREFECT_DEPLOYMENT="$PREFECT_DEPLOYMENT"

echo
echo "== 2/4 Smoke-test configured model routes and prompt contract =="
poetry run python scripts/smoke_model_routes.py \
  --case-config "$CASE_CONFIG" \
  --stage writer \
  --stage critic
poetry run python scripts/smoke_prompt_contract.py \
  --case-config "$CASE_CONFIG" \
  --stage writer

echo
echo "== 3/4 Run one smoke document =="
smoke_watch_args=()
if [[ "$WATCH_SMOKE" == "true" ]]; then
  smoke_watch_args=(--watch --watch-interval "$SMOKE_WATCH_INTERVAL" --watch-timeout "$SMOKE_WATCH_TIMEOUT")
fi
poetry run prefect deployment run "synthetic-ner-generation/$PREFECT_DEPLOYMENT" \
  --flow-run-name "$SMOKE_RUN_NAME" \
  --params "{\"case_config\":\"$CASE_CONFIG\",\"template\":\"$TEMPLATE\",\"documents\":$SMOKE_DOCUMENTS,\"review_scenario\":false,\"review_entities\":false}" \
  "${smoke_watch_args[@]}"

echo
echo "== 4/4 Schedule ${DOCUMENTS}-document run =="
schedule_args=()
if [[ -n "$START_IN" ]]; then
  schedule_args=(--start-in "$START_IN")
fi
poetry run prefect deployment run "synthetic-ner-generation/$PREFECT_DEPLOYMENT" \
  --flow-run-name "$RUN_NAME" \
  --params "{\"case_config\":\"$CASE_CONFIG\",\"template\":\"$TEMPLATE\",\"documents\":$DOCUMENTS,\"review_scenario\":false,\"review_entities\":false}" \
  "${schedule_args[@]}"

echo
echo "Queued Prefect runs. Watch them at http://localhost:4200/runs"
