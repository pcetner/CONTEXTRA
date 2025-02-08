import unittest
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_manager.tokens import ConversationToken, ConversationState

class TestTokens(unittest.TestCase):

    def test_create_conversation_token(self):
        token = ConversationToken(text="hello")
        self.assertEqual(token.text, "hello")
        self.assertEqual(token.p_unimportant, 0.5)
        self.assertEqual(token.frequency, 0)
        self.assertEqual(token.last_used_turn, 0)
        self.assertFalse(token.user_marked_important)

    def test_conversation_state(self):
        state = ConversationState()
        self.assertEqual(len(state.tokens), 0)
        self.assertEqual(state.turn_count, 0)

        # Add a token
        token = ConversationToken(text="world")
        state.tokens.append(token)
        self.assertEqual(len(state.tokens), 1)

if __name__ == "__main__":
    unittest.main()
