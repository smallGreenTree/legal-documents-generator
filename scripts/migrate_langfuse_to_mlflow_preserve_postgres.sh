#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PLATFORM_ENV="${PLATFORM_ENV:-.env.platform}"
OVERRIDE_FILE="${PLATFORM_VOLUME_OVERRIDE:-docker-compose.postgres-volume.override.yml}"
PREFECT_DB_NAME="${LEGACY_PREFECT_DB_NAME:-prefect}"
BACKUP_ROOT="${MIGRATION_BACKUP_ROOT:-migration-backups}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$BACKUP_ROOT/langfuse-to-mlflow-$TIMESTAMP"

log() {
  printf '[langfuse-to-mlflow] %s\n' "$*"
}

fail() {
  printf '[langfuse-to-mlflow] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

container_env() {
  local container="$1"
  local key="$2"
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container" \
    | awk -v key="$key" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }'
}

find_postgres_container() {
  if [[ -n "${LEGACY_POSTGRES_CONTAINER:-}" ]]; then
    docker inspect "$LEGACY_POSTGRES_CONTAINER" >/dev/null 2>&1 \
      || fail "LEGACY_POSTGRES_CONTAINER does not exist: $LEGACY_POSTGRES_CONTAINER"
    printf '%s\n' "$LEGACY_POSTGRES_CONTAINER"
    return
  fi

  local candidates count
  candidates="$(docker ps -aq --filter label=com.docker.compose.service=postgres)"
  count="$(printf '%s\n' "$candidates" | awk 'NF {count++} END {print count+0}')"
  if [[ "$count" -ne 1 ]]; then
    fail "Found $count Compose PostgreSQL containers. Set LEGACY_POSTGRES_CONTAINER explicitly."
  fi
  printf '%s\n' "$candidates"
}

update_platform_env() {
  export MIGRATION_ENV_PATH="$PLATFORM_ENV"
  export MIGRATION_POSTGRES_USER="$legacy_postgres_user"
  export MIGRATION_POSTGRES_PASSWORD="$legacy_postgres_password"
  export MIGRATION_POSTGRES_DB="$legacy_postgres_db"
  export MIGRATION_PREFECT_DB="$PREFECT_DB_NAME"
  export MIGRATION_POSTGRES_VOLUME="$postgres_volume"
  export MIGRATION_PROJECT_NAME="$legacy_project"

  python3 <<'PY'
import os
from pathlib import Path

path = Path(os.environ["MIGRATION_ENV_PATH"])
text = path.read_text(encoding="utf-8")
updates = {
    "PLATFORM_PROJECT_NAME": os.environ["MIGRATION_PROJECT_NAME"],
    "POSTGRES_DATA_VOLUME": os.environ["MIGRATION_POSTGRES_VOLUME"],
    "POSTGRES_ADMIN_USER": os.environ["MIGRATION_POSTGRES_USER"],
    "POSTGRES_ADMIN_PASSWORD": os.environ["MIGRATION_POSTGRES_PASSWORD"],
    "POSTGRES_ADMIN_DB": os.environ["MIGRATION_POSTGRES_DB"],
    "PREFECT_DB_USER": os.environ["MIGRATION_POSTGRES_USER"],
    "PREFECT_DB_PASSWORD": os.environ["MIGRATION_POSTGRES_PASSWORD"],
    "PREFECT_DB_NAME": os.environ["MIGRATION_PREFECT_DB"],
}


def dotenv_value(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise SystemExit("PostgreSQL environment values must not contain newlines")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$")
    return f'"{escaped}"'


remaining = dict(updates)
lines = []
for line in text.splitlines():
    key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
    if key in remaining:
        lines.append(f"{key}={dotenv_value(remaining.pop(key))}")
    else:
        lines.append(line)
if remaining:
    if lines and lines[-1]:
        lines.append("")
    lines.extend(f"{key}={dotenv_value(value)}" for key, value in remaining.items())
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
path.chmod(0o600)
PY

  unset MIGRATION_ENV_PATH MIGRATION_POSTGRES_USER MIGRATION_POSTGRES_PASSWORD
  unset MIGRATION_POSTGRES_DB MIGRATION_PREFECT_DB MIGRATION_POSTGRES_VOLUME
  unset MIGRATION_PROJECT_NAME
}

remove_langfuse_containers() {
  local service container_ids
  for service in langfuse-worker langfuse-web clickhouse minio redis; do
    container_ids="$(
      docker ps -aq \
        --filter "label=com.docker.compose.project=$legacy_project" \
        --filter "label=com.docker.compose.service=$service"
    )"
    if [[ -n "$container_ids" ]]; then
      log "Removing obsolete $service container(s); data volumes are retained."
      docker rm -f $container_ids >/dev/null
    fi
  done
}

require_command docker
require_command python3
require_command make
docker info >/dev/null 2>&1 || fail "Docker is not available."
[[ -f "$PLATFORM_ENV" ]] \
  || fail "Missing $PLATFORM_ENV. Copy .env.platform.example and configure MLflow passwords first."
if grep -Eq '^[A-Z0-9_]*PASSWORD=replace-me$' "$PLATFORM_ENV"; then
  fail "Replace all placeholder MLflow/platform passwords in $PLATFORM_ENV before migration."
fi

postgres_container="$(find_postgres_container)"
legacy_project="$(docker inspect --format '{{index .Config.Labels "com.docker.compose.project"}}' "$postgres_container")"
[[ -n "$legacy_project" && "$legacy_project" != '<no value>' ]] \
  || fail "The selected PostgreSQL container has no Compose project label."

mount_details="$(
  docker inspect --format \
    '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{printf "%s|%s" .Type .Name}}{{end}}{{end}}' \
    "$postgres_container"
)"
mount_type="${mount_details%%|*}"
postgres_volume="${mount_details#*|}"
[[ "$mount_type" == "volume" && -n "$postgres_volume" ]] \
  || fail "PostgreSQL /var/lib/postgresql/data is not backed by a named Docker volume."

legacy_postgres_user="$(container_env "$postgres_container" POSTGRES_USER)"
legacy_postgres_password="$(container_env "$postgres_container" POSTGRES_PASSWORD)"
legacy_postgres_db="$(container_env "$postgres_container" POSTGRES_DB)"
[[ -n "$legacy_postgres_user" && -n "$legacy_postgres_password" && -n "$legacy_postgres_db" ]] \
  || fail "Could not read the existing PostgreSQL initialization identity."

if [[ "$(docker inspect --format '{{.State.Running}}' "$postgres_container")" != "true" ]]; then
  log "Starting the existing PostgreSQL container for validation and backup."
  docker start "$postgres_container" >/dev/null
fi

log "Validating Prefect history in database '$PREFECT_DB_NAME'."
prefect_runs_before="$(
  docker exec "$postgres_container" \
    psql -X -U "$legacy_postgres_user" -d "$PREFECT_DB_NAME" -Atqc \
    'SELECT count(*) FROM flow_run;'
)" || fail "Cannot read Prefect flow_run from the existing PostgreSQL volume."
[[ "$prefect_runs_before" =~ ^[0-9]+$ ]] || fail "Unexpected Prefect flow-run count."
log "Existing Compose project: $legacy_project"
log "Existing PostgreSQL volume: $postgres_volume"
log "Existing Prefect flow runs: $prefect_runs_before"

mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"
log "Backing up every PostgreSQL database before changing containers."
docker exec "$postgres_container" pg_dumpall -U "$legacy_postgres_user" \
  > "$BACKUP_DIR/postgres-all.sql"
docker exec "$postgres_container" \
  pg_dump -U "$legacy_postgres_user" -Fc "$PREFECT_DB_NAME" \
  > "$BACKUP_DIR/prefect.dump"
chmod 600 "$BACKUP_DIR/postgres-all.sql" "$BACKUP_DIR/prefect.dump"
[[ -s "$BACKUP_DIR/postgres-all.sql" && -s "$BACKUP_DIR/prefect.dump" ]] \
  || fail "Database backup is empty; no containers were removed."
docker inspect "$postgres_container" > "$BACKUP_DIR/postgres-container-inspect.json"
chmod 600 "$BACKUP_DIR/postgres-container-inspect.json"

if [[ "${MIGRATION_YES:-0}" != "1" ]]; then
  printf 'Type MIGRATE to remove Langfuse containers and start Prefect + MLflow: '
  read -r confirmation
  [[ "$confirmation" == "MIGRATE" ]] || fail "Migration cancelled; backups were retained."
fi

if [[ -f .env.langfuse ]]; then
  log "Archiving .env.langfuse outside the active configuration."
  mv .env.langfuse "$BACKUP_DIR/env.langfuse"
  chmod 600 "$BACKUP_DIR/env.langfuse"
fi

log "Pinning the platform to existing PostgreSQL volume: $postgres_volume"
update_platform_env
cat > "$OVERRIDE_FILE" <<EOF
services:
  postgres:
    volumes:
      - platform_postgres_data:/var/lib/postgresql/data

volumes:
  platform_postgres_data:
    external: true
    name: $postgres_volume
EOF

remove_langfuse_containers

log "Starting PostgreSQL, Prefect Server, and MLflow with the preserved database volume."
make PLATFORM_ENV="$PLATFORM_ENV" PLATFORM_VOLUME_OVERRIDE="$OVERRIDE_FILE" platform-up

new_postgres_container="$(
  docker compose --env-file "$PLATFORM_ENV" \
    -f docker-compose.platform.yml -f "$OVERRIDE_FILE" ps -q postgres
)"
[[ -n "$new_postgres_container" ]] || fail "The migrated PostgreSQL container is not running."
actual_volume="$(
  docker inspect --format \
    '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' \
    "$new_postgres_container"
)"
[[ "$actual_volume" == "$postgres_volume" ]] \
  || fail "Safety check failed: PostgreSQL started with '$actual_volume', expected '$postgres_volume'."

prefect_runs_after="$(
  docker exec "$new_postgres_container" \
    psql -X -U "$legacy_postgres_user" -d "$PREFECT_DB_NAME" -Atqc \
    'SELECT count(*) FROM flow_run;'
)"
[[ "$prefect_runs_after" =~ ^[0-9]+$ ]] || fail "Could not validate Prefect after migration."
if (( prefect_runs_after < prefect_runs_before )); then
  fail "Prefect history check failed: before=$prefect_runs_before after=$prefect_runs_after"
fi

log "Migration succeeded. Prefect flow runs preserved: $prefect_runs_after"
log "Backup directory: $BACKUP_DIR"
log "Old Langfuse data volumes were intentionally retained for rollback."
log "Next: run 'make prefect-up' to register this repository's deployments and start its worker."
