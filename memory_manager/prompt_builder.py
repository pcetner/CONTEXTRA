from memory_manager.tokens import ConversationState

def build_prompt(state: ConversationState, user_input: str) -> str:
    """
    Constructs the final prompt by concatenating retained tokens (in chronological order)
    followed by the user's new input. Adjust format for your LLM's style.
    """
    # Sort retained tokens by their original insertion order if needed.
    # (Assuming they are appended in order, or store a creation index.)
    # For simplicity, let's assume state.tokens is already chronological.
    
    # Join all token texts to form 'history':
    history_text = " ".join(token.text for token in state.tokens)
    
    # Some models might need explicit role labels, e.g.:
    #   <|user|> last user message
    #   <|assistant|> ...
    # Adjust as your model’s best practice.
    # We'll keep it simple here:
    prompt = (
        f"{history_text}\n"
        f"User: {user_input}\n"
        f"System:"
    )
    return prompt
