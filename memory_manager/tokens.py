from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class ConversationToken:
    """
    Represents either a single token or a chunk of text in the conversation.
    """
    text: str
    frequency: int = 1
    last_used_turn: int = 0
    user_importance: float = 0.0
    p_unimportant: float = 0.5
    embedding: Optional[np.ndarray] = None

class ConversationState:
    def __init__(self):
        self.tokens: List[ConversationToken] = []
        self.turn_count: int = 0
        self.important_chunks: List[str] = []  # Store user-marked important text chunks
