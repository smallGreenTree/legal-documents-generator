#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"

if [[ -d "$SCRIPT_DIR/../ner_platform_workspace/synthetic_dataset_NER" ]]; then
  DEFAULT_SOURCE_WORKSPACE="$(cd "$SCRIPT_DIR/../ner_platform_workspace" && pwd -P)"
elif [[ -d "$SCRIPT_DIR/../../synthetic_dataset_NER" ]]; then
  DEFAULT_SOURCE_WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
else
  DEFAULT_SOURCE_WORKSPACE=""
fi

usage() {
  cat <<EOF
Usage:
  $SCRIPT_NAME --destination PATH [--source-workspace PATH]

Creates a clean, non-Git candidate snapshot of the NER platform.

Safety properties:
  - never changes a source repository
  - never copies .git history or configures a remote
  - never commits or pushes
  - refuses dirty source repositories
  - refuses an existing destination or audit directory
  - excludes local configuration, generated output, archives, SBOMs, and CI
  - writes provenance and scan results beside, not inside, the snapshot

Defaults:
  --source-workspace  $DEFAULT_SOURCE_WORKSPACE

Example:
  $SCRIPT_NAME --destination /private/tmp/ner-platform-eppo-candidate
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '%s\n' "$*"
}

SOURCE_WORKSPACE="$DEFAULT_SOURCE_WORKSPACE"
DESTINATION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-workspace)
      [[ $# -ge 2 ]] || die "--source-workspace requires a path"
      SOURCE_WORKSPACE="$2"
      shift 2
      ;;
    --destination)
      [[ $# -ge 2 ]] || die "--destination requires a path"
      DESTINATION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$DESTINATION" ]] || die "--destination is required"
[[ -d "$SOURCE_WORKSPACE" ]] || die "source workspace does not exist: $SOURCE_WORKSPACE"

SOURCE_WORKSPACE="$(cd "$SOURCE_WORKSPACE" && pwd -P)"

destination_parent="$(dirname "$DESTINATION")"
destination_name="$(basename "$DESTINATION")"
[[ -d "$destination_parent" ]] || die "destination parent does not exist: $destination_parent"
DESTINATION="$(cd "$destination_parent" && pwd -P)/$destination_name"
AUDIT_DIR="${DESTINATION}.audit"

[[ ! -e "$DESTINATION" ]] || die "destination already exists: $DESTINATION"
[[ ! -e "$AUDIT_DIR" ]] || die "audit directory already exists: $AUDIT_DIR"

case "$DESTINATION" in
  "$SOURCE_WORKSPACE"|"$SOURCE_WORKSPACE"/*)
    die "destination must be outside the source workspace"
    ;;
esac

command -v git >/dev/null 2>&1 || die "git is required"
command -v rg >/dev/null 2>&1 || die "ripgrep (rg) is required"

source_repositories=(
  "synthetic_dataset_NER"
  "ner_evaluation_pipeline"
  "ner-prefect-pipeline"
  "ner-rules"
  "synthetic_dataset_NER_infra"
)

destination_components=(
  "synthetic-data-generator"
  "evaluation-pipeline"
  "llm-ner-pipeline"
  "rules-ner"
  "infrastructure"
)

dirty_count=0
missing_count=0

note "Checking source repositories..."
for index in 0 1 2 3 4; do
  repository_path="$SOURCE_WORKSPACE/${source_repositories[$index]}"
  if [[ ! -d "$repository_path/.git" ]]; then
    printf '  MISSING  %s\n' "$repository_path" >&2
    missing_count=$((missing_count + 1))
    continue
  fi

  if [[ -n "$(git -C "$repository_path" status --porcelain)" ]]; then
    printf '  DIRTY    %s\n' "$repository_path" >&2
    git -C "$repository_path" status --short >&2
    dirty_count=$((dirty_count + 1))
  else
    printf '  CLEAN    %s\n' "$repository_path"
  fi
done

[[ $missing_count -eq 0 ]] || die "one or more required repositories are missing"
[[ $dirty_count -eq 0 ]] || die "resolve or deliberately commit source changes before exporting"

mkdir "$DESTINATION"
mkdir "$AUDIT_DIR"

MANIFEST="$AUDIT_DIR/source-manifest.txt"
EXCLUSIONS="$AUDIT_DIR/excluded-files.txt"
SYMLINKS="$AUDIT_DIR/skipped-symlinks.txt"
REFERENCE_FINDINGS="$AUDIT_DIR/personal-reference-findings.txt"
SECRET_FINDINGS="$AUDIT_DIR/high-confidence-secret-findings.txt"

: >"$MANIFEST"
: >"$EXCLUSIONS"
: >"$SYMLINKS"
: >"$REFERENCE_FINDINGS"
: >"$SECRET_FINDINGS"

is_excluded() {
  local path="$1"

  case "$path" in
    .env|.env.*|*/.env|*/.env.*)
      return 0
      ;;
    .github|.github/*|*/.github|*/.github/*)
      return 0
      ;;
    .idea|.idea/*|*/.idea|*/.idea/*|.vscode|.vscode/*|*/.vscode|*/.vscode/*)
      return 0
      ;;
    .venv|.venv/*|*/.venv|*/.venv/*|__pycache__|__pycache__/*|*/__pycache__|*/__pycache__/*)
      return 0
      ;;
    .prefect|.prefect/*|*/.prefect|*/.prefect/*|.pytest_cache|.pytest_cache/*|*/.pytest_cache|*/.pytest_cache/*)
      return 0
      ;;
    output|output/*|*/output|*/output/*|outputs|outputs/*|*/outputs|*/outputs/*)
      return 0
      ;;
    tmp|tmp/*|*/tmp|*/tmp/*|traces|traces/*|*/traces|*/traces/*)
      return 0
      ;;
    memory|memory/*|*/memory|*/memory/*|migration-backups|migration-backups/*|*/migration-backups|*/migration-backups/*)
      return 0
      ;;
    config_case/generated|config_case/generated/*|*/config_case/generated|*/config_case/generated/*)
      return 0
      ;;
    sbom|sbom/*|*/sbom|*/sbom/*|sboms|sboms/*|*/sboms|*/sboms/*)
      return 0
      ;;
    *.zip|*.7z|*.tar|*.tar.gz|*.tgz|*.cdx.json|*.DS_Store)
      return 0
      ;;
  esac

  return 1
}

note "Exporting tracked files into a non-Git candidate snapshot..."
for index in 0 1 2 3 4; do
  source_name="${source_repositories[$index]}"
  component_name="${destination_components[$index]}"
  repository_path="$SOURCE_WORKSPACE/$source_name"
  component_path="$DESTINATION/$component_name"
  source_commit="$(git -C "$repository_path" rev-parse HEAD)"
  copied_count=0
  excluded_count=0

  mkdir "$component_path"

  while IFS= read -r -d '' relative_path; do
    if is_excluded "$relative_path"; then
      printf '%s/%s\n' "$source_name" "$relative_path" >>"$EXCLUSIONS"
      excluded_count=$((excluded_count + 1))
      continue
    fi

    source_file="$repository_path/$relative_path"
    destination_file="$component_path/$relative_path"

    if [[ -L "$source_file" ]]; then
      printf '%s/%s\n' "$source_name" "$relative_path" >>"$SYMLINKS"
      continue
    fi

    [[ -f "$source_file" ]] || continue
    mkdir -p "$(dirname "$destination_file")"
    cp -p "$source_file" "$destination_file"
    copied_count=$((copied_count + 1))
  done < <(git -C "$repository_path" ls-files -z)

  {
    printf 'component=%s\n' "$component_name"
    printf 'source_repository=%s\n' "$source_name"
    printf 'source_commit=%s\n' "$source_commit"
    printf 'copied_files=%s\n' "$copied_count"
    printf 'excluded_files=%s\n\n' "$excluded_count"
  } >>"$MANIFEST"

  printf '  %-26s %4d files\n' "$component_name" "$copied_count"
done

note "Scanning for personal repository and workstation references..."
personal_reference_pattern='smallGreenTree|evaluation-NER-pipeline|legal-documents-generator|ner-llm-pipeline|ner-infrastructure|Panoss-Mac-Studio|/Users/antonis-antono|[A-Za-z0-9._%+-]+@gmail\.com'
rg -n -i --hidden "$personal_reference_pattern" "$DESTINATION" >"$REFERENCE_FINDINGS" || true

note "Running a high-confidence local secret-pattern scan..."
high_confidence_secret_pattern='BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{20,}|sk-[A-Za-z0-9]{32,}'
rg -l -i --hidden "$high_confidence_secret_pattern" "$DESTINATION" >"$SECRET_FINDINGS" || true

scan_failed=0

if [[ -s "$REFERENCE_FINDINGS" ]]; then
  note "REVIEW REQUIRED: personal or prior-repository references were found."
  note "  $REFERENCE_FINDINGS"
  scan_failed=1
fi

if [[ -s "$SECRET_FINDINGS" ]]; then
  note "REVIEW REQUIRED: possible high-confidence secrets were found."
  note "  $SECRET_FINDINGS"
  scan_failed=1
fi

if [[ -s "$SYMLINKS" ]]; then
  note "REVIEW REQUIRED: symlinks were skipped rather than followed."
  note "  $SYMLINKS"
  scan_failed=1
fi

if command -v gitleaks >/dev/null 2>&1; then
  note "Running Gitleaks against the candidate snapshot..."
  if ! gitleaks dir "$DESTINATION" \
    --no-banner \
    --redact \
    --report-format json \
    --report-path "$AUDIT_DIR/gitleaks.json"; then
    note "REVIEW REQUIRED: Gitleaks reported findings."
    note "  $AUDIT_DIR/gitleaks.json"
    scan_failed=1
  fi
else
  note "REVIEW REQUIRED: Gitleaks is not installed; run an approved full secret scanner before transfer."
  scan_failed=1
fi

cat >"$AUDIT_DIR/README.txt" <<EOF
This directory is intentionally outside the candidate snapshot.

Candidate snapshot:
  $DESTINATION

The candidate contains no .git directory and this script did not initialize,
commit, configure a remote, or push anything.

Before transfer:
  1. Review every file in the candidate snapshot.
  2. Review source-manifest.txt and excluded-files.txt.
  3. Resolve every personal-reference and secret-scan finding.
  4. Review synthetic documents and annotations for real personal or case data.
  5. Add sanitized environment examples and EPPO-approved CI separately.
  6. Run EPPO-approved secret, dependency, container, and data scans.
  7. Copy the approved candidate into a separately cloned EPPO repository.
  8. Commit and push only after institutional review.
EOF

note ""
note "Candidate snapshot: $DESTINATION"
note "Local audit report: $AUDIT_DIR"
note "No source repository was changed. Nothing was committed or pushed."

if [[ $scan_failed -ne 0 ]]; then
  note "The snapshot was created, but it is NOT cleared for transfer. Review the audit findings."
  exit 2
fi

note "Automated checks passed. Manual and institutional review are still required."
