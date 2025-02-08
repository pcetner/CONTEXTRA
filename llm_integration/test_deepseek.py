# test_deepseek.py

import unittest
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_integration.deepseek import generate_with_deepseek

class TestDeepseekIntegration(unittest.TestCase):

    def test_generate_with_deepseek_basic(self):
        """
        Tests if generate_with_deepseek successfully calls the model via ollama
        and returns a non-empty string.
        """
        test_prompt = "Hello, deepseek. How are you? Respond in 20 words or less."
        
        # Call the function
        response = generate_with_deepseek(prompt=test_prompt)

        # Basic assertions
        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertTrue(len(response.strip()) > 0, "Response should not be empty.")

        # Optional: Check for certain keywords, or partial match
        # self.assertIn("some keyword", response.lower())

if __name__ == "__main__":
    unittest.main()
