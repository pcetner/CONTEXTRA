import math
from memory_manager.tokens import ConversationState
from typing import List
import numpy as np

def compute_semantic_importance(token, state: ConversationState) -> float:
    if not token.embedding is not None:
        return 0.5
    
    # Compare with other tokens' embeddings
    similarities = []
    for other in state.tokens:
        if other != token and other.embedding is not None:
            similarity = np.dot(token.embedding, other.embedding) / (
                np.linalg.norm(token.embedding) * np.linalg.norm(other.embedding)
            )
            similarities.append(similarity)
    
    return max(similarities) if similarities else 0.5

def update_probabilities(state: ConversationState):
    current_turn = state.turn_count
    max_turns = max(t.last_used_turn for t in state.tokens) if state.tokens else current_turn
    
    for token in state.tokens:
        # Recency factor (0 to 1, where 1 is oldest)
        recency = (current_turn - token.last_used_turn) / max(1, max_turns)
        
        # Frequency factor (inverse, where higher frequency = lower value)
        frequency = 1 / (1 + token.frequency)
        
        # User importance factor (0 to 1, where 1 is least important)
        importance = 1 - (token.user_importance / 10)
        
        # Semantic importance based on embeddings
        semantic_importance = compute_semantic_importance(token, state)
        
        # Combine factors
        token.p_unimportant = (
            0.3 * recency +
            0.2 * frequency +
            0.3 * importance +
            0.2 * (1 - semantic_importance)
        )
