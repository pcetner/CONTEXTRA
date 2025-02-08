import unittest
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_manager.tokens import ConversationToken, ConversationState
from memory_manager.scorer import update_probabilities

class TestScorer(unittest.TestCase):

    def test_update_probabilities_basic(self):
        state = ConversationState()
        # Turn count = 0 initially
        t1 = ConversationToken(text="hello", frequency=0, last_used_turn=0, user_marked_important=False)
        t2 = ConversationToken(text="world", frequency=2, last_used_turn=0, user_marked_important=False)
        state.tokens.extend([t1, t2])

        # Call the scorer
        update_probabilities(state)

        # We expect each token to have p_unimportant updated from 0.5.
        self.assertNotEqual(t1.p_unimportant, 0.5)
        self.assertNotEqual(t2.p_unimportant, 0.5)

    def test_update_probabilities_importance_flags(self):
        state = ConversationState()
        # Suppose we simulate a user_marked_important token
        imp_token = ConversationToken(text="IMPORTANT", user_marked_important=True)
        reg_token = ConversationToken(text="regular", user_marked_important=False)
        state.tokens.extend([imp_token, reg_token])

        update_probabilities(state)

        # We expect the important token to have a lower p_unimportant (because it's more important).
        self.assertLess(imp_token.p_unimportant, reg_token.p_unimportant)

if __name__ == "__main__":
    unittest.main()
