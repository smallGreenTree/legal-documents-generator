# Prefect End-to-End Project Map

This map shows how the project runs from entrypoint to final artifacts, with each Prefect-visible stage tied back to the first relevant Python line.

## High-Level Flow

`main.py` is the local alias entrypoint for normal CLI runs:

```python
from src.synthetic_ner.cli import main
```

The Prefect entrypoint is `src/synthetic_ner/prefect_pipeline.py`, which exposes two flows:

- `synthetic-ner-generation`: generate a full synthetic legal NER document.
- `synthetic-ner-document-quality`: score an existing generated document without regenerating it.

## Generation Flow Map

The outer Prefect flow is the manager-facing map:

```python
@flow(name="synthetic-ner-generation")
def generate_dataset(
```

Inside that flow, one document generation run maps one by one to these stages.

| Order | Prefect stage | First relevant Python line | What it does | Main downstream code |
| ---: | --- | --- | --- | --- |
| 1 | `scenario-selection` | `@task(name="scenario-selection")` | Resolves the selected scenario, effective doc type, fraud type, document count, optional source schema, and input-file list. | `_publish_scenario_artifacts(scenario)` |
| 2 | `human-scenario-review` | `@task(name="human-scenario-review")` | Optional pause where a human can approve, reload, alter, or cancel the scenario before config ingestion. | `pause_flow_run(...)` |
| 3 | `configs-ingestion` | `@task(name="configs-ingestion")` | Loads `.env`, root config, case config, model routing, paths, workflow settings, section targets, and template environment. | `build_runtime_context(args, project_root)` in `engine.py` |
| 4 | `faker-entities` | `@task(name="faker-entities")` | Generates or normalizes defendants, collateral people, charged organisations, associated organisations, metadata, and indictment counts. | `resolve_document_inputs(context)` in `engine.py` |
| 5 | `human-entity-review` | `@task(name="human-entity-review")` | Optional pause where a human can approve, cancel, or paste edited document-input JSON before schema and document creation. | `pause_flow_run(...)` |
| 6 | `select-doc-id` | `@task(name="select-doc-id")` | Scans output, partial output, schema, and memory roots to select the next safe document id. | `_used_doc_counters(context)` and `make_doc_id(...)` |
| 7 | `case-schema` | `@task(name="case-schema")` | Builds or loads the case relationship schema for the selected document id. | `resolve_schema_for_document(...)` in `engine.py` |
| 8 | `langgraph-langfuse-generation` | `@task(name="langgraph-langfuse-generation")` | Runs the inner LangGraph workflow and passes the Prefect flow run id into Langfuse trace metadata. | `run_document_graph(...)` in `tasks/orchestrator.py` |
| 9 | `end-of-pipeline-file-audit` | `@task(name="end-of-pipeline-file-audit")` | Collects generated files, memory files, schema, partials, checksums, timestamps, and publishes Prefect artifacts. | `_collect_document_files(...)` and `_publish_prefect_artifacts(...)` |

The actual loop in the Prefect flow is:

```python
for document_index in range(context.documents):
```

That loop calls the stages in order:

```python
scenario = select_scenario(
```

If `review_scenario=True`, the flow pauses here:

```python
scenario = review_selected_scenario(
```

```python
context = ingest_configs(
```

```python
document = resolve_entities(context)
```

If `review_entities=True`, the flow pauses here:

```python
document = review_document_entities(
```

```python
selected_doc_id = select_doc_id(context)
```

```python
doc_id, schema = build_case_schema(
```

```python
run_langgraph_langfuse(context, document, schema, doc_id)
```

```python
audit_created_files(context, doc_id)
```

## Stage Details

### 1. Scenario Selection

First relevant line:

```python
@task(name="scenario-selection")
```

The task resolves the manager-visible scenario before loading the full runtime context:

```python
selected_doc_type = doc_type or profile.get("doc_type")
```

```python
selected_fraud_type = fraud_type or profile.get("fraud_type")
```

It publishes a table artifact of the concrete input files:

```python
_publish_scenario_artifacts(scenario)
```

The input-file artifact includes:

- root config
- selected scenario config
- workflow prompts
- document template
- optional source schema
- optional quality scoring config
- present `.env` and `.env.langfuse` files

Each row records role, path, required/optional status, existence, size, modified time, and SHA-256 checksum when available.

### 2. Human Scenario Review

First relevant line:

```python
@task(name="human-scenario-review")
```

This optional task is enabled with `review_scenario=True` or `--review-scenario`.

It pauses the flow here:

```python
response = pause_flow_run(
```

The review input supports:

- `action="continue"`: proceed with the selected scenario.
- `action="reload"`: rebuild the scenario from reviewed fields such as `case_config`, `documents`, `doc_type`, `fraud_type`, or `from_schema`.
- `action="cancel"`: stop the run before config ingestion.

### 3. Configs Ingestion

First relevant line:

```python
@task(name="configs-ingestion")
```

The task creates a runtime context. That context is the shared object passed through the rest of the pipeline.

Important first lines underneath:

```python
load_env_files(project_root)
```

```python
context = build_runtime_context(args, project_root)
```

`build_runtime_context` starts here:

```python
def build_runtime_context(args: Namespace, project_root: Path) -> RuntimeContext:
```

It resolves:

- `config.yaml`
- selected `config_case/*.yaml`
- document count
- document type and fraud type
- output, schema, and memory directories
- Jinja template environment
- section word targets
- prose overrides
- optional source schema path

### 4. Faker Entities

First relevant line:

```python
@task(name="faker-entities")
```

The task delegates to:

```python
document = resolve_document_inputs(context)
```

`resolve_document_inputs` starts here:

```python
def resolve_document_inputs(context: RuntimeContext) -> DocumentInputs:
```

It resolves:

- people through `resolve_case_entities(...)`
- case metadata through `resolve_case_metadata(...)`
- indictment counts through `resolve_counts(...)`

The output is a `DocumentInputs` object, not just Faker data. It is the concrete cast and legal metadata used by schema generation, memory creation, prose generation, template rendering, and ground truth output.

### 5. Human Entity Review

First relevant line:

```python
@task(name="human-entity-review")
```

This optional task is enabled with `review_entities=True` or `--review-entities`.

It first publishes the current resolved document inputs:

```python
_publish_document_inputs_artifact(
```

Then it pauses the flow:

```python
response = pause_flow_run(
```

The review input supports:

- `action="continue"`: use the generated cast unchanged.
- `action="apply_json"`: parse the edited `document_json` and continue with those people, organisations, metadata, and counts.
- `refresh_counts=True`: recompute indictment counts after edits so deleted or altered defendants/organisations are reflected.
- `action="cancel"`: stop before document id selection, schema creation, and generation.

### 6. Select Document Id

First relevant line:

```python
@task(name="select-doc-id")
```

The stage starts from existing artifacts:

```python
used_counters = _used_doc_counters(context)
```

Then it selects the next id:

```python
doc_id = make_doc_id(context.doc_type, context.fraud_type, next_counter)
```

This is important because it now considers all known roots:

- `output/{doc_id}`
- `output/_partial/{doc_id}`
- `schemas/{doc_id}.json`
- `memory/case_{doc_id}`

That prevents a new Prefect run from accidentally mixing fresh output with stale partials, stale schema files, or old memory.

### 7. Case Schema

First relevant line:

```python
@task(name="case-schema")
```

The task delegates to:

```python
doc_id, schema = resolve_schema_for_document(
```

`resolve_schema_for_document` starts here:

```python
def resolve_schema_for_document(
```

It has two paths:

- If `--from-schema` is provided, load and normalize the source schema under a new target `doc_id`.
- Otherwise, auto-generate or config-normalize a schema for the current concrete cast.

The auto-generated relationship graph starts here:

```python
def make_case_schema(
```

The schema is the factual relationship layer. It records persons, organisations, and edges such as control, instruction, conspiracy, and funds flow.

### 8. LangGraph and Langfuse Generation

First relevant line:

```python
@task(name="langgraph-langfuse-generation")
```

This task captures the Prefect flow run id:

```python
prefect_flow_run_id = str(get_run_context().flow_run.id)
```

Then it runs the inner generation graph:

```python
run_document_graph(
```

`run_document_graph` starts here:

```python
def run_document_graph(
```

It creates:

- `TraceStore` for Langfuse/local trace metadata.
- `CaseMemoryManager` and initial `CASE_MEMORY.md`.
- planner, writer, and critic model clients.
- prompt objects from Langfuse Prompt Management or config fallback.
- the compiled LangGraph document workflow.

The graph is created here:

```python
graph = build_document_graph(
```

The graph is invoked here:

```python
final_state = graph.invoke(
```

## Inner LangGraph Map

The inner graph starts with:

```python
def build_document_graph(
```

The workflow class starts here:

```python
class DocumentWorkflow:
```

The compiled graph is built here:

```python
def build_graph(self):
```

The registered nodes are:

| Order | LangGraph node | First relevant Python line | What it does |
| ---: | --- | --- | --- |
| 1 | `document_planner` | `builder.add_node(` | Produces the whole-document plan from memory, doc type, fraud type, case number, and section order. |
| 2 | `process_sections` | `builder.add_node(` | Plans, writes, polishes, critiques, validates, revises, repairs, and stores each section. |
| 3 | `render_document` | `builder.add_node(` | Validates complete section outputs, renders the template, saves files, and writes the generation report. |

The edges are linear:

```python
builder.add_edge(START, "document_planner")
```

```python
builder.add_edge("document_planner", "process_sections")
```

```python
builder.add_edge("process_sections", "render_document")
```

```python
builder.add_edge("render_document", END)
```

### Document Planner Node

First relevant line:

```python
def document_planner_node(self, state: WorkflowState) -> WorkflowState:
```

It delegates to:

```python
document_plan = self.planner.plan_document(
```

The planner method starts here:

```python
def plan_document(
```

The output is appended to case memory and stored in the graph instruction channel.

### Process Sections Node

First relevant line:

```python
def process_sections_node(self, state: WorkflowState) -> WorkflowState:
```

It groups sections with dependency awareness:

```python
for group in _parallel_section_groups(section_order):
```

The dependency rules start here:

```python
def _parallel_section_groups(section_order: list[str]) -> list[list[str]]:
```

Current dependencies:

- `facts` waits for `history` and `charges`.
- `evidence` waits for `facts`.
- `assessment` waits for `facts`.

Each section runs through:

```python
results = [self._run_section_workflow(state, group[0])]
```

or, when safe:

```python
executor.submit(
```

### Section Workflow

First relevant line:

```python
def _run_section_workflow(
```

One section follows this inner sequence:

| Order | Step | First relevant Python line | What it does |
| ---: | --- | --- | --- |
| 1 | Section contract | `section_contract = build_section_contract(section_name)` | Loads deterministic requirements for the section. |
| 2 | Section plan | `section_plan = self.planner.plan_section(` | Produces a section-specific plan. |
| 3 | Writer | `section_text = self.writer.write_section(` | Writes chunked section content. |
| 4 | Cleaner | `section_text = clean_generated_section_text(section_text)` | Removes markdown, metadata, thinking tags, repetition, and broken endings. |
| 5 | Critic | `review = self.critic.review_section(` | Gets structured critic edits and risk signals. |
| 6 | Validator | `validator_issues = validate_section_text(` | Applies deterministic grounding, style, repetition, and entity checks. |
| 7 | Revision | `section_text = self.writer.write_section(` | Rewrites when critic or validator issues remain and revision budget allows. |
| 8 | Final repair/fallback | `final_text, final_issues = self._finalize_section_text(` | Repairs deterministic issues or replaces with fallback prose. |

### Writer and Polisher

First writer line:

```python
def write_section(
```

The writer loops until the section word target is reached:

```python
while words_so_far < word_target:
```

The first model call asks the writer for structured JSON:

```python
result = self.client.invoke(
```

The writer packet is parsed here:

```python
writer_packet = parse_writer_packet(result.text)
```

The second model call polishes the JSON content into prose:

```python
polished_result = self.client.invoke(
```

Each chunk is saved as a partial artifact:

```python
self._write_partial_section(
```

This is why `output/_partial/{doc_id}/sections/{section}/r{revision}/` is useful for debugging.

### Critic

First critic line:

```python
def review_section(
```

It builds compact prompts from memory, section context, contract, section plan, and section text:

```python
user_prompt = render_prompt_template(
```

It then invokes the critic model:

```python
result = self.client.invoke(
```

The structured critic result is parsed here:

```python
return self._parse_result(result.text)
```

If the model times out or has a provider error, the critic degrades gracefully and relies on deterministic validation.

### Final Section Validation

First relevant line:

```python
def _finalize_section_text(
```

The finalizer always reads current memory:

```python
memory_text = self.memory_manager.read_memory(self.memory_path)
```

It validates final text:

```python
final_issues = validate_section_text(
```

If validation still fails, it builds deterministic fallback text:

```python
fallback_text = build_deterministic_fallback_section(
```

### Render Document

First relevant line:

```python
def render_document_node(self, state: WorkflowState) -> WorkflowState:
```

Before rendering, it checks for missing, placeholder, or too-short sections:

```python
problems = collect_section_output_problems(
```

Then it renders the final Jinja template:

```python
rendered_text = render_document_text(self.context, self.document, ordered_sections)
```

Then it saves the final document artifacts:

```python
save_document_artifacts(
```

The save function starts here:

```python
def save_document_artifacts(
```

It writes:

- `schemas/{doc_id}.json`
- `output/{doc_id}/{doc_id}.txt`
- `output/{doc_id}/groundtruth.tsv`

The generation report is written here:

```python
write_generation_report(
```

## End-of-Pipeline File Audit

First relevant line:

```python
@task(name="end-of-pipeline-file-audit")
```

It creates:

```python
audit_path = context.output_dir / doc_id / "file_audit.json"
```

It collects known document artifacts:

```python
files = _collect_document_files(context, doc_id, exclude={audit_path})
```

Each file record includes:

- path
- size in bytes
- SHA-256 checksum
- modification timestamp

The collected manifest is also published to Prefect as markdown and table artifacts.

## Quality Scoring Flow Map

The second Prefect flow scores an existing document:

```python
@flow(name="synthetic-ner-document-quality")
```

It first resolves and publishes the selected scenario and input files:

```python
scenario = select_scenario(
```

Then it loads the same runtime context:

```python
context = ingest_configs(
```

Then delegates to the scoring task:

```python
return score_document_quality(context, doc_id, quality_config)
```

The scoring task starts here:

```python
@task(name="score-document-quality")
```

The quality report builder starts here:

```python
def build_quality_report(
```

It scores the latest partial revision for each section:

```python
section = _score_section(
```

It reuses deterministic validation:

```python
issues = validate_section_text(
```

It publishes:

- Prefect table artifact: section scores.
- Prefect markdown artifact: quality summary.

## Artifact Map

| Artifact | First relevant Python line | Producer | Purpose |
| --- | --- | --- | --- |
| `memory/case_{doc_id}/CASE_MEMORY.md` | `memory_path = memory_manager.create_initial_memory(` | `run_document_graph` | Human-readable source of truth for the case run. |
| `output/_partial/{doc_id}/sections/...` | `self._write_partial_section(` | `SectionWriter.write_section` | Debuggable chunk and revision history. |
| `schemas/{doc_id}.json` | `write_case_schema(schema_path, schema)` | `save_document_artifacts` | Relationship graph and factual backbone. |
| `output/{doc_id}/{doc_id}.txt` | `txt_path.write_text(rendered_text, encoding="utf-8")` | `save_document_artifacts` | Final rendered legal document. |
| `output/{doc_id}/groundtruth.tsv` | `write_groundtruth(gt_path, gt_rows)` | `save_document_artifacts` | NER gold rows for entities, references, dates, amounts, addresses, VAT, and controls. |
| `output/{doc_id}/generation_report.md` | `write_generation_report(` | `render_document_node` | Planner, contract, review, revision, and trace summary. |
| `output/{doc_id}/file_audit.json` | `audit_path = context.output_dir / doc_id / "file_audit.json"` | `audit_created_files` | Manifest with file sizes, timestamps, and hashes. |
| Prefect markdown/table artifacts | `create_markdown_artifact(` | `prefect_pipeline.py` helpers | UI-visible config, schema, memory, output, quality, and created-file summaries. |

## Current Mapping Assessment

The project now maps cleanly to Prefect at the outer orchestration level:

- Config ingestion is visible.
- Entity generation is visible.
- Document id selection is visible.
- Case schema creation is visible.
- LangGraph/Langfuse generation is visible.
- Final file audit is visible.
- Existing-document quality scoring is exposed as a separate Prefect flow.

The main intentional compression is that the whole inner planner-writer-polisher-critic-validator loop appears as one Prefect task, `langgraph-langfuse-generation`. That is a reasonable split because LangGraph and Langfuse already expose the inner node/call-level detail.

If the goal is a fully expanded Prefect UI for non-engineers, the next mapping improvement would be to split quality scoring into the generation flow after `end-of-pipeline-file-audit`, or to expose per-section Prefect tasks. If the goal is reliable generation debugging, the current map is better: Prefect handles run-level orchestration while LangGraph/Langfuse handle detailed generation internals.
