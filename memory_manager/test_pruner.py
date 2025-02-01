import unittest
from tokens import ConversationToken, ConversationState
from pruner import prune_if_needed

class TestPruner(unittest.TestCase):

    def test_prune_if_needed_below_limit(self):
        state = ConversationState()
        # Add 3 tokens
        state.tokens.append(ConversationToken(text="t1", p_unimportant=0.8))
        state.tokens.append(ConversationToken(text="t2", p_unimportant=0.2))
        state.tokens.append(ConversationToken(text="t3", p_unimportant=0.5))

        max_tokens = 5  # We have only 3, so no pruning is needed
        prune_if_needed(state, max_tokens)
        self.assertEqual(len(state.tokens), 3, "No token should be pruned if under limit")

    def test_prune_if_needed_over_limit(self):
        state = ConversationState()
        # Add 5 tokens with varying p_unimportant
        tokens = [
            ConversationToken(text="t1", p_unimportant=0.8),
            ConversationToken(text="t2", p_unimportant=0.1),
            ConversationToken(text="t3", p_unimportant=0.9),
            ConversationToken(text="t4", p_unimportant=0.3),
            ConversationToken(text="t5", p_unimportant=0.7),
        ]
        state.tokens.extend(tokens)

        max_tokens = 2  # We'll need to prune 3 tokens
        prune_if_needed(state, max_tokens)

        self.assertLessEqual(len(state.tokens), 2, "Should prune down to 2 tokens or fewer")

        # Check that the pruned tokens are likely the ones with highest p_unimportant
        remaining_p_unimportant = [t.p_unimportant for t in state.tokens]
        self.assertFalse(any(p > 0.8 for p in remaining_p_unimportant),
                         "Tokens with highest p_unimportant (>=0.8) should be removed first")

if __name__ == "__main__":
    unittest.main()
