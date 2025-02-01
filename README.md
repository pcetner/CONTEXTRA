C – Context

O – Oriented

N – Neural

T – Token

EX – EXclusion

TR – TRimming

A – Adaptation

- **`llm_integration/deepseek.py`**: Handles calling **deepseek-r1:70b** via `ollama`.
- **`memory_manager/tokens.py`**: Defines data classes (e.g., `ConversationToken`, `ConversationState`).
- **`memory_manager/scorer.py`**: Functions to update token importance/unimportance probabilities.
- **`memory_manager/pruner.py`**: Logic to decide and remove tokens (probabilistic eviction).
- **`memory_manager/prompt_builder.py`**: Constructs the final prompt from retained tokens and user’s latest message.
- **`main.py`**: Ties everything together. Runs a loop, accepts user input, updates memory, calls the model, etc.
- **`requirements.txt`**: Lists Python dependencies
