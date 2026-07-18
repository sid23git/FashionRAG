"""Query expansion module for fashion text retrieval."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expands fashion-related queries using a predefined synonym dictionary.
    
    Attributes:
        expansion_dict (Dict[str, List[str]]): A mapping of terms to their expanded forms.
    """
    
    def __init__(self) -> None:
        """Initializes the QueryExpander with predefined mappings."""
        self.expansion_dict: Dict[str, List[str]] = {
            "shirt": ["shirt", "apparel", "clothing", "top"],
            "tshirt": ["tshirt", "tee", "top", "clothing"],
            "jeans": ["jeans", "denim", "pants", "trousers"],
            "dress": ["dress", "gown", "outfit"],
            "formal": ["formal", "business", "office", "professional"],
            "casual": ["casual", "everyday", "relaxed"],
            "jacket": ["jacket", "coat", "outerwear"],
            "hoodie": ["hoodie", "sweatshirt"],
            "sneakers": ["sneakers", "shoes", "trainers"],
        }
        logger.info("QueryExpander initialized with %d mappings.", len(self.expansion_dict))

    def expand(self, query: str) -> str:
        """Expands the query based on the predefined dictionary.
        
        Args:
            query: The original text query.
            
        Returns:
            A single string containing the expanded query.
        """
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            if word in self.expansion_dict:
                expanded_words.extend(self.expansion_dict[word])
            else:
                expanded_words.append(word)
                
        # Remove duplicates while preserving order
        seen = set()
        final_words = []
        for w in expanded_words:
            if w not in seen:
                seen.add(w)
                final_words.append(w)
                
        expanded_query = " ".join(final_words)
        logger.info("Expanded query: '%s' -> '%s'", query, expanded_query)
        return expanded_query
