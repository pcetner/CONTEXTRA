import random
import sys

# Local imports
from memory_manager.tokens import ConversationToken, ConversationState
from memory_manager.scorer import update_probabilities
from memory_manager.pruner import prune_if_needed
from memory_manager.prompt_builder import build_prompt
from llm_integration.deepseek import generate_with_deepseek

def main():
    # Seed random for reproducibility (optional)
    random.seed(42)

    # For demonstration, let's keep a small token limit
    MAX_TOKENS = 50

    state = ConversationState()

    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # 1) Increment turn counter
        state.turn_count += 1
        
        # 2) Tokenize or chunk user input (very naive approach)
        tokens = user_input.split()
        for t in tokens:
            new_token = ConversationToken(
                text=t,
                frequency=1,
                last_used_turn=state.turn_count
            )
            state.tokens.append(new_token)

        # 3) Update probabilities
        update_probabilities(state)

        # 4) Prune if needed
        prune_if_needed(state, MAX_TOKENS)

        # 5) Build prompt
        final_prompt = build_prompt(state, user_input)

        # 6) Generate with deepseek-r1:8b (via ollama)
        response = generate_with_deepseek(final_prompt)

        # 7) Print response
        print(f"AI: {response.strip()}")

        # 8) Optional: parse the response to detect references, adjust frequencies, etc.
        # Skipping that for now.

if __name__ == "__main__":
    main()
