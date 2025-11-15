"""
Simple embedding model for semantic caching.

Uses TF-IDF for lightweight semantic similarity without external dependencies.
Can be replaced with OpenAI embeddings or other models later.
"""

import re
import math
from typing import List, Dict
from collections import Counter, defaultdict


class SimpleEmbeddings:
    """
    Lightweight TF-IDF based embeddings for semantic similarity.

    This is a simple implementation that doesn't require external APIs.
    For production, consider using:
    - OpenAI embeddings
    - Sentence transformers
    - Cohere embeddings
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.vocabulary: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.document_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        tf = {}
        for token, count in token_counts.items():
            tf[token] = count / total_tokens

        return tf

    def _compute_idf(self, documents: List[str]):
        """
        Compute inverse document frequency.

        This should be called during initialization with a corpus of sample queries.
        """
        doc_count = len(documents)
        token_doc_count = defaultdict(int)

        # Count documents containing each token
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                token_doc_count[token] += 1

        # Compute IDF
        self.idf_scores = {}
        for token, count in token_doc_count.items():
            self.idf_scores[token] = math.log(doc_count / (1 + count))

        self.document_count = doc_count

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Returns a fixed-size vector of TF-IDF scores.
        """
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)

        # Create sparse TF-IDF vector
        tfidf = {}
        for token, tf_score in tf.items():
            idf_score = self.idf_scores.get(token, 0.0)
            tfidf[token] = tf_score * idf_score

        # Convert to dense vector
        # Use hash trick to map tokens to fixed dimension
        embedding = [0.0] * self.embedding_dim

        for token, score in tfidf.items():
            # Simple hash to index
            idx = hash(token) % self.embedding_dim
            embedding[idx] += score

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def __call__(self, text: str) -> List[float]:
        """Make the embedder callable"""
        return self.embed(text)


def create_default_embeddings() -> SimpleEmbeddings:
    """
    Create embeddings model with pre-trained on common queries.
    """
    embedder = SimpleEmbeddings(embedding_dim=128)

    # Sample corpus of common queries for IDF calculation
    sample_queries = [
        # Jira queries
        "show my jira tickets",
        "list my tasks",
        "get my issues",
        "show open tickets",
        "list assigned to me",
        "create a new ticket",
        "update ticket status",
        "close the issue",
        "assign ticket to user",
        "add comment to ticket",
        "search for bugs",
        "show high priority tickets",
        "list tickets in project",

        # GitHub queries
        "show my pull requests",
        "list my PRs",
        "show open issues",
        "list repositories",
        "create pull request",
        "merge the PR",
        "close issue",
        "list branches",
        "show commits",
        "search code",
        "list contributors",

        # Slack queries
        "show recent messages",
        "list channels",
        "get notifications",
        "send message",
        "create channel",
        "list users",
        "search messages",
        "show threads",

        # General
        "help",
        "status",
        "what can you do",
        "hello",
        "goodbye",
        "thanks",
    ]

    # Compute IDF scores
    embedder._compute_idf(sample_queries)

    return embedder
