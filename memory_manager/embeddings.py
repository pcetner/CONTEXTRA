from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from .tokens import ConversationToken

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def compute_embedding(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def update_token_embeddings(self, tokens: List[ConversationToken]):
        for token in tokens:
            if token.embedding is None:
                token.embedding = self.compute_embedding(token.text) 