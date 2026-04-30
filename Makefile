.PHONY: sync-langfuse

sync-langfuse:
	poetry run python -m src.synthetic_ner.sync_langfuse_prompts --commit-message "$(or $(MSG),Sync prompt templates)"
