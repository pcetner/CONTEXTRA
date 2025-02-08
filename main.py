import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Local imports
from memory_manager.tokens import ConversationToken, ConversationState
from memory_manager.scorer import update_probabilities
from memory_manager.pruner import prune_if_needed
from memory_manager.prompt_builder import build_prompt
from llm_integration.deepseek import generate_with_deepseek
import threading
import time

def thinking_animation(stop_event):
    """Display a thinking animation with dots"""
    print("\nThinking", end="", flush=True)
    while not stop_event.is_set():
        print(".", end="", flush=True)
        time.sleep(1)

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

        # 6) Generate with deepseek-r1:8b (via ollama) with streaming
        # Start thinking animation in a separate thread
        stop_thinking = threading.Event()
        thinking_thread = threading.Thread(target=thinking_animation, args=(stop_thinking,))
        thinking_thread.daemon = True
        thinking_thread.start()

        # Generate response
        response_received = False
        response = []
        for token in generate_with_deepseek(final_prompt, stream=True):
            if not response_received:
                # Clear the thinking animation line
                stop_thinking.set()  # Signal the animation to stop
                thinking_thread.join()  # Wait for the thread to finish
                print("\r" + " " * 50 + "\r", end="", flush=True)
                print("AI: ", end="", flush=True)
                response_received = True
            print(token, end="", flush=True)
            response.append(token)
        
        if not response_received:
            print("\rNo response received from the model.")
        print()  # New line after response is complete

        # 8) Optional: parse the response to detect references, adjust frequencies, etc.
        # Skipping that for now.

if __name__ == "__main__":
    main()
