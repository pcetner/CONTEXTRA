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
from transformers import AutoTokenizer
from scipy.spatial.distance import cosine
import numpy as np
from memory_manager.embeddings import EmbeddingManager
from memory_manager.command_parser import parse_user_input

def thinking_animation(stop_event):
    """Display a thinking animation with dots"""
    print("\nThinking", end="", flush=True)
    while not stop_event.is_set():
        print(".", end="", flush=True)
        time.sleep(1)

def main():
    # Initialize components
    random.seed(42)
    MAX_TOKENS = 50
    state = ConversationState()
    embedding_manager = EmbeddingManager()
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")

    print("Type 'exit' to quit. Commands: !important <text>, !forget <text>, !summary\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Parse commands
        command, text = parse_user_input(user_input)
        
        if command == 'important':
            state.important_chunks.append(text)
            print("Marked as important!")
            continue
        elif command == 'summary':
            print("\nImportant information:")
            for chunk in state.important_chunks:
                print(f"- {chunk}")
            continue
        else:
            text = user_input
        
        # 1) Increment turn counter
        state.turn_count += 1
        
        # 2) Tokenize input
        token_ids = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Process tokens
        for t in tokens:
            existing_token = next((tok for tok in state.tokens if tok.text == t), None)
            if existing_token:
                existing_token.frequency += 1
                existing_token.last_used_turn = state.turn_count
            else:
                new_token = ConversationToken(
                    text=t,
                    frequency=1,
                    last_used_turn=state.turn_count
                )
                state.tokens.append(new_token)
        
        # 3) Update embeddings
        embedding_manager.update_token_embeddings(state.tokens)
        
        # 4) Update probabilities
        update_probabilities(state)
        
        # 5) Prune if needed
        prune_if_needed(state, MAX_TOKENS)
        
        # 6) Build prompt
        final_prompt = build_prompt(state, text)
        
        # 7) Generate response
        stop_thinking = threading.Event()
        thinking_thread = threading.Thread(target=thinking_animation, args=(stop_thinking,))
        thinking_thread.daemon = True
        thinking_thread.start()

        response_received = False
        response = []
        for token in generate_with_deepseek(final_prompt, stream=True):
            if not response_received:
                stop_thinking.set()
                thinking_thread.join()
                print("\r" + " " * 50 + "\r", end="", flush=True)
                print("AI: ", end="", flush=True)
                response_received = True
            print(token, end="", flush=True)
            response.append(token)
        
        if not response_received:
            print("\rNo response received from the model.")
        print()

        # 8) Analyze response and update token importance
        response_text = ''.join(response)
        response_embedding = embedding_manager.compute_embedding(response_text)
        
        # Update importance based on similarity to response
        for token in state.tokens:
            if token.embedding is not None:
                similarity = embedding_manager.compute_similarity(token.embedding, response_embedding)
                token.user_importance = max(token.user_importance, similarity * 5)  # Scale to 0-5 range

if __name__ == "__main__":
    main()
