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

SCENARIO_RUNS=(
  "financial-fraud|config_case/case_1.yaml|templates/en_indictment.j2|indictment|financial_fraud"
  "procurement-fraud|config_case/case_1.yaml|templates/en_indictment.j2|indictment|procurement_fraud"
  "eu-subsidy-fraud|config_case/case_1.yaml|templates/en_indictment.j2|indictment|eu_subsidy_fraud"
  "customs-evasion|config_case/case_1.yaml|templates/en_indictment.j2|indictment|customs_evasion"
  "money-laundering|config_case/case_1.yaml|templates/en_indictment.j2|indictment|money_laundering"
  "tax-evasion|config_case/case_1.yaml|templates/en_indictment.j2|indictment|tax_evasion"
  "bribery|config_case/case_1.yaml|templates/en_indictment.j2|indictment|bribery"
  "insider-trading|config_case/case_1.yaml|templates/en_indictment.j2|indictment|insider_trading"
  "financial-fraud-variant|config_case/case_1_variant_2.yaml|templates/en_indictment.j2|indictment|financial_fraud"
  "evidence-money-laundering|tests/functional/test_scenarios/test_evidence/money_laundering.yaml|tests/functional/test_scenarios/test_evidence/en_evidence_test.j2|indictment|money_laundering"
)

export PREFECT_HOME PREFECT_API_URL

echo "== Apple Studio synthetic NER run =="
echo "smoke_case_config=$CASE_CONFIG"
echo "smoke_template=$TEMPLATE"
echo "scenario_runs=$DOCUMENTS"
echo "prefect_api=$PREFECT_API_URL"
echo

echo "== 1/4 Deploy Prefect generation flow =="
make _prefect-deploy \
  CASE_CONFIG="$CASE_CONFIG" \
  TEMPLATE="$TEMPLATE" \
  DOCS="1" \
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
echo "== 4/4 Schedule ${DOCUMENTS} scenario run(s) =="
schedule_args=()
if [[ -n "$START_IN" ]]; then
  schedule_args=(--start-in "$START_IN")
fi
if (( DOCUMENTS > ${#SCENARIO_RUNS[@]} )); then
  echo "Requested DOCUMENTS=$DOCUMENTS but only ${#SCENARIO_RUNS[@]} scenario runs are configured." >&2
  exit 1
fi

for ((index = 0; index < DOCUMENTS; index++)); do
  IFS="|" read -r scenario_name scenario_case scenario_template scenario_doc_type scenario_fraud_type \
    <<< "${SCENARIO_RUNS[$index]}"
  run_number=$(printf "%02d" "$((index + 1))")
  params=$(printf '{"case_config":"%s","template":"%s","documents":1,"doc_type":"%s","fraud_type":"%s","review_scenario":false,"review_entities":false}' \
    "$scenario_case" "$scenario_template" "$scenario_doc_type" "$scenario_fraud_type")

  echo "queue ${run_number}/${DOCUMENTS}: ${scenario_name}"
  poetry run prefect deployment run "synthetic-ner-generation/$PREFECT_DEPLOYMENT" \
    --flow-run-name "${RUN_NAME}-${run_number}-${scenario_name}" \
    --params "$params" \
    "${schedule_args[@]}"
done

echo
echo "Queued Prefect runs. Watch them at http://localhost:4200/runs"
