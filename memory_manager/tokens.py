from dataclasses import dataclass, field
from typing import List

@dataclass
class ConversationToken:
    """
    Represents either a single token or a chunk of text in the conversation.
    """
    text: str
    p_unimportant: float = 0.5
    frequency: int = 0
    last_used_turn: int = 0
    user_marked_important: bool = False

@dataclass
class ConversationState:
    """
    Holds the list of tokens and a turn counter to track conversation turns.
    """
    tokens: List[ConversationToken] = field(default_factory=list)
    turn_count: int = 0
