# test_deepseek.py

import unittest
from deepseek import generate_with_deepseek

class TestDeepseekIntegration(unittest.TestCase):

    def test_generate_with_deepseek_basic(self):
        """
        Tests if generate_with_deepseek successfully calls the model via ollama
        and returns a non-empty string.
        """
        test_prompt = "Hello, deepseek. How are you?"
        
        # Call the function
        response = generate_with_deepseek(prompt=test_prompt)

        # Basic assertions
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertTrue(len(response.strip()) > 0, "Response should not be empty.")

        # Optional: Check for certain keywords, or partial match
        # self.assertIn("some keyword", response.lower())

if __name__ == "__main__":
    unittest.main()
