import math
from memory_manager.tokens import ConversationState

def update_probabilities(state):
    """
    Updates each token's probability of being unimportant (p_unimportant).
    This example uses recency and frequency. You can add more signals.
    """
    for token in state.tokens:
        # Normalize frequency to a [0,1] range (very rough example)
        freq_norm = min(token.frequency / 5.0, 1.0)

        # Compute a recency score: tokens used more recently are more important.
        recency_distance = (state.turn_count - token.last_used_turn)
        recency_score = math.exp(-0.2 * recency_distance)

        # If user explicitly marked token as important, give a big boost.
        user_flag_score = 1.0 if token.user_marked_important else 0.0

        # Combine signals into an "importance" metric (simple sum in this example).
        importance = freq_norm + recency_score + 3.0 * user_flag_score
        
        # Convert importance to an unimportance probability in [0,1].
        # p_unimportant = 1 - sigmoid(importance)
        # We'll use a straightforward logistic transform for illustration:
        logistic_value = 1 / (1 + math.exp(-importance))
        token.p_unimportant = 1 - logistic_value
