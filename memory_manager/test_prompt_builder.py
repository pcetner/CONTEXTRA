import unittest
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_manager.tokens import ConversationToken, ConversationState
from memory_manager.prompt_builder import build_prompt

class TestPromptBuilder(unittest.TestCase):

    def test_build_prompt(self):
        state = ConversationState()
        state.tokens.append(ConversationToken(text="Hello"))
        state.tokens.append(ConversationToken(text="World"))

        user_input = "How are you?"
        prompt = build_prompt(state, user_input)

        # Check basic structure
        self.assertIn("Hello World", prompt, "Should contain concatenated token texts")
        self.assertIn("How are you?", prompt, "Should include user input")

if __name__ == "__main__":
    unittest.main()
