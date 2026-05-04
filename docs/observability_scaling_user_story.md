# Observability and Scaling User Story

## User Story

As a developer of the Synthetic Legal NER Generator, I want detailed per-node observability and run-level performance summaries, so that I can identify slow, expensive, truncated, or hallucination-prone stages before changing the architecture or scaling to larger documents.

## Problem Statement

The current workflow can become slow and difficult to debug as documents grow. The likely bottlenecks include too many LLM calls, large prompts, Qwen thinking consuming output budget, critic and revision loops, repeated full `CASE_MEMORY.md` usage, and local model latency.

Without structured observability, it is hard to know whether quality problems come from prompt size, output truncation, bad context selection, critic behavior, revision loops, or model limitations.

## Acceptance Criteria

1. Each LLM call records enough metadata to diagnose performance and quality issues.

2. Per-call metadata includes:
   - stage
   - section name
   - revision round
   - task id
   - prompt character count
   - response character count
   - prompt token count when available
   - response token count when available
   - latency in milliseconds
   - model name
   - output budget
   - context window setting
   - `done_reason`
   - whether the final response was empty

3. The workflow writes a run-level summary after each generated document.

4. The run summary includes:
   - total LLM calls
   - total latency
   - latency by stage
   - token usage by stage
   - largest prompt
   - largest response
   - slowest call
   - sections that triggered revisions
   - calls that ended with `done_reason: length`
   - calls that returned an empty response

5. Langfuse traces expose the same key metadata so debugging can happen visually.

6. The local trace artifact clearly identifies bottleneck candidates without requiring manual inspection of every raw payload.

7. The workflow can compare small and larger document runs using the same summary fields.

8. The summary helps decide whether the next fix should be smaller context packets, fewer LLM calls, larger output budgets, larger context windows, deterministic templates, or critic changes.

## Definition of Done

- A generated document produces a readable local performance summary.
- Langfuse shows per-call metadata sufficient to diagnose truncation, empty responses, and slow stages.
- A developer can identify the slowest node and largest prompt from one summary file.
- Empty responses and `done_reason: length` are explicitly counted.
- The summary supports before/after comparison when optimizing context size or workflow structure.

## Proposed Implementation Order

1. Extend the trace metadata for every LLM call with prompt size, response size, token counts, latency, model options, and empty-response status.
2. Aggregate per-call trace records into a document-level run summary.
3. Save the summary under the document trace directory.
4. Add stage-level totals for planner, writer, critic, revision, and validation-related calls.
5. Highlight bottleneck candidates such as largest prompt, slowest call, truncations, and empty responses.
6. Use the summary from several runs to decide where section-specific context packets and deterministic templates will save the most time.
