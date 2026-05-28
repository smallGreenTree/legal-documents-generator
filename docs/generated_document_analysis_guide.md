# Generated Document Analysis Guide

Use this guide to judge whether a generated synthetic legal document is acceptable, risky, or unusable.

## Quick Verdict Scale

- **Good:** Facts are grounded, sections are coherent, no invented amounts/dates/entities, and revisions improve the text.
- **Acceptable with fixes:** Mostly grounded, but has repetition, weak legal style, or minor unsupported wording.
- **Bad:** Contains invented material, repeated sections, contradictory details, or critic revisions that do not improve the output.
- **Unusable:** Final document cannot be trusted because it mixes old artifacts, invents material facts, or omits required entities.

## Current Document Snapshot

The current partial document is **bad but diagnostically useful**.

It is not hopeless: the entity names, broad case structure, and some procedural framing are present. But it should not be accepted as a final synthetic indictment yet.

Main observed problems:

- The `facts` section invents specific invoice counts and amounts such as `12 invoices`, `15 invoices`, `18 invoices`, `EUR 1.2 million`, `EUR 1.5 million`, and `EUR 2.1 million`.
- Several sections repeat the same narrative instead of progressing.
- Some revisions are identical in word count, which suggests the revision loop may be rewriting mechanically or not applying critic feedback.
- The `history` section repeats its opening investigation narrative twice.
- The critic allowed sections with clear repeated fragments.
- `assessment` and `evidence` are only at `r0` in the current partials, while other sections reached `r2`.

## What To Check First

### 1. Grounding

Ask: **Can every factual detail be traced to case memory, schema, or allowed prompt context?**

Fail examples:

- New monetary amounts not present in source facts.
- New invoice numbers.
- New search dates.
- New roles such as `chief executive officer`, `compliance officer`, or `accountant` if not explicitly in the source.
- Claims that recipients were misled, paid money, or suffered loss without source support.

Score:

- `5`: every fact is grounded.
- `3`: core facts are grounded, but wording adds mild assumptions.
- `1`: material invented facts appear.

### 2. Repetition

Ask: **Does each paragraph add new information, or does it restate the same entity map?**

Warning signs:

- Same defendant/company list appears in multiple paragraphs.
- Same sentence structure repeats across chunks.
- Final combined section reads like chunk concatenation rather than one section.

Score:

- `5`: no obvious repetition.
- `3`: some repetition, still readable.
- `1`: repeated blocks dominate the section.

### 3. Legal Style

Ask: **Does it sound like an indictment, or like a generic fraud essay?**

Good signs:

- Neutral, formal language.
- Clear counts and particulars.
- No dramatic wording.
- Uses exact legal charge names.

Bad signs:

- “complex financial operation”
- “fraudulent activity escalated”
- “meticulously managed”
- “continued undetected”
- unsupported intent language.

Score:

- `5`: legal/procedural and controlled.
- `3`: readable but generic.
- `1`: journalistic or speculative.

### 4. Section Purpose

Ask: **Does each section do its own job?**

Expected roles:

- `persons`: identify people only.
- `companies`: identify organisations and relationships only.
- `history`: investigation timeline and procedural events.
- `charges`: counts, defendants, charge periods, legal basis.
- `facts`: factual narrative of alleged conduct.
- `evidence`: evidential items and what they support.
- `assessment`: legal/prosecutorial assessment, not new facts.

Fail example:

- `history` retells the whole fraud scheme instead of focusing on investigation steps.
- `facts` repeats persons/companies sections.
- `assessment` invents conclusions not grounded in facts.

### 5. Revision Quality

Ask: **Did `r1` and `r2` actually improve the section?**

Check:

- Compare word count between rounds.
- Compare repeated phrases.
- Check whether critic issues disappeared.
- Check whether new hallucinations were introduced.

Red flag:

- `r0`, `r1`, and `r2` have identical word counts or near-identical content.

## Minimum Acceptance Criteria

A generated document should pass these checks before being treated as good output:

- No ungrounded invoice amounts, invoice counts, dates, names, addresses, roles, or legal claims.
- No paragraph-level repetition.
- Every section has a distinct purpose.
- Critic output names specific edits, and the next revision visibly applies them.
- Final document is assembled from the latest accepted revision of each section.
- Run artifacts come only from the current `doc_id`.

## Recommended Manual Review Order

1. Check `CASE_MEMORY.md` and schema first.
2. Read `facts` because it has the highest hallucination risk.
3. Read `charges` because legal mistakes are highest impact.
4. Read `history` for invented procedural events.
5. Read `evidence` for invented exhibits or unsupported proof claims.
6. Read `assessment` last and verify it does not introduce new facts.

## Decision Rule

Reject the document if any of these are true:

- It invents monetary values, invoice numbers, or transaction counts.
- It contains repeated paragraphs caused by chunk stitching.
- It uses old artifacts from a previous run.
- It reaches `r2` but still contains critic-reported issues.
- The final output is missing while partials exist.

For the current run, the main rejection reason is:

**Material hallucination plus repetition in the `facts` and `history` sections.**
