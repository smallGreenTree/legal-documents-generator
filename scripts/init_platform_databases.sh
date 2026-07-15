#!/usr/bin/env bash
set -euo pipefail

psql --set=ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=admin_user="$POSTGRES_USER" --set=admin_password="$POSTGRES_PASSWORD" <<'SQL'
SELECT format('ALTER ROLE %I WITH LOGIN PASSWORD %L', :'admin_user', :'admin_password')\gexec
SQL

create_database() {
  local database_name="$1"
  local database_user="$2"
  local database_password="$3"

  psql --set=ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    --set=db_user="$database_user" --set=db_password="$database_password" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'db_user', :'db_password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'db_user')\gexec
SELECT format('ALTER ROLE %I WITH LOGIN PASSWORD %L', :'db_user', :'db_password')\gexec
SQL
  psql --set=ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    --set=db_name="$database_name" --set=db_user="$database_user" <<'SQL'
SELECT format('CREATE DATABASE %I OWNER %I', :'db_name', :'db_user')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'db_name')\gexec
SQL
}

create_database "$PREFECT_DB_NAME" "$PREFECT_DB_USER" "$PREFECT_DB_PASSWORD"
create_database "$MLFLOW_DB_NAME" "$MLFLOW_DB_USER" "$MLFLOW_DB_PASSWORD"
