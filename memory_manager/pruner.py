import random
from memory_manager.tokens import ConversationState

def prune_if_needed(state: ConversationState, max_token_count: int):
    """
    If the number of tokens in state.tokens exceeds max_token_count,
    probabilistically remove the tokens with the highest p_unimportant.
    """
    while len(state.tokens) > max_token_count:
        # Sort tokens by p_unimportant (descending).
        state.tokens.sort(key=lambda t: t.p_unimportant, reverse=True)
        # Attempt to remove the most unimportant token first.
        candidate = state.tokens[0]
        
        # Try random eviction based on p_unimportant.
        if random.random() < candidate.p_unimportant:
            state.tokens.pop(0)
        else:
            # If the random draw doesn't remove it, forcibly remove it anyway
            # to ensure we eventually free enough space.
            state.tokens.pop(0)

