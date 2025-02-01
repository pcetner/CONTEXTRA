import unittest
from tokens import ConversationToken, ConversationState
from prompt_builder import build_prompt

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
