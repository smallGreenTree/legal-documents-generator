# Productisation User Story

## User Story

As a user of the Synthetic Legal NER Generator, I want the project to be plug-and-play, efficient with local LLMs, transparent in its memory handling, and configurable for controlled entity variance, so that I can generate high-quality synthetic legal documents for rule-based NER testing without manually debugging every run.

## Acceptance Criteria

1. The repository includes a clear README that explains setup, configuration, running, debugging, Langfuse syncing, output locations, and common failure modes.

2. A new user can run the project locally with clear commands for installing dependencies, starting Ollama and Langfuse, syncing prompts, generating documents, and inspecting outputs and traces.

3. The generator keeps `CASE_MEMORY.md` as the human-readable source of truth, but does not pass the entire memory to every LLM call by default.

4. Planner, writer, and critic receive section-specific compact context packets containing only the facts needed for their task.

5. The exact context packet sent to each LLM call is visible in Langfuse traces.

6. Token and context controls are fully configurable from `config.yaml`, including writer, planner, critic, and Ollama context settings.

7. Critique behavior is configurable, with support for strict exact-entity checking, allowed entity surface forms, intentional name or company variation, typo or diacritic variance, and abbreviation or reference-form variance.

8. The critic distinguishes between invalid hallucinations and configured allowed variance.

9. Entity generation supports canonical identities plus controlled surface forms, so a person or company can appear in multiple valid ways while preserving gold-label identity.

10. The final generated document is suitable for rule-based NER evaluation, with known canonical entities, allowed variants, optional controlled mistakes, and traceable provenance.

## Definition of Done

- A fresh user can clone the repo, follow the README, run one generation, and understand where every artifact is.
- A full run completes without empty LLM responses caused by hidden token limits.
- Langfuse shows the prompt, compact context, model response, critic scores, and relevant metadata.
- Config changes, not code edits, control model budgets, context limits, entity variance, and critic behavior.
- Generated output can be audited back to `CASE_MEMORY.md` and the runtime context packets.

## Proposed Implementation Order

1. Write the README and local run guide.
2. Add section-specific context packets.
3. Add `ollama.num_ctx` and finish config-driven token controls.
4. Add configurable entity variance and noise policy.
5. Teach the critic to score both correctness and configured noise.
6. Run one full generated document and inspect Langfuse traces stage by stage.
