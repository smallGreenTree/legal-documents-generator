#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${repo_root}/sbom"
sbom_file="${output_dir}/synthetic-dataset-ner.cdx.json"
cyclonedx_version="7.3.1"

mkdir -p "${output_dir}"

uvx --from "cyclonedx-bom==${cyclonedx_version}" cyclonedx-py poetry \
  "${repo_root}" \
  --no-dev \
  --mc-type application \
  --output-reproducible \
  --output-format JSON \
  --spec-version 1.6 \
  --validate \
  --output-file "${sbom_file}"

echo "Generated ${sbom_file}"
