# src/text_processor.py
"""Text processing functionality for social media analysis."""

import re
from typing import List, Dict, Any, Set, Tuple
import pandas as pd
from collections import Counter
import logging
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)


class TextProcessor:
    """Process and analyze social media text content."""

    def __init__(self):
        """Initialize text processor."""
        # Load NLTK stopwords for multiple languages
        from nltk.corpus import stopwords
        self.stopwords = {
            'english': set(stopwords.words('english')),
            'arabic': set(stopwords.words('arabic'))
        }

        # Expo-related keywords
        self.EXPO_KEYWORDS = {
            'expo', 'expo2020', 'إكسبو', 'اكسبو', 'expo2020dubai', 'dubai2020',
            'worldexpo', 'dubaiexpo', 'معرض', 'دبي', 'expo2020', 'expodubai'
        }

        # Compile regex patterns
        self._hashtag_pattern = re.compile(r'#\w+')
        self._mention_pattern = re.compile(r'@\w+')
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self._emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        self._arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
        self._latin_pattern = re.compile(r'[a-zA-Z]+')

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all text processing steps to a dataframe.

        Args:
            df: The input dataframe with a 'message' column.

        Returns:
            The dataframe with new text-based feature columns.
        """
        logger.info("Processing dataframe with TextProcessor...")
        df_processed = df.copy()

        # Ensure 'message' column is string and handle NaNs
        text_series = df_processed['message'].astype(str).fillna('')

        # --- Feature Creation ---
        df_processed['text_length'] = text_series.str.len()
        df_processed['word_count'] = text_series.apply(lambda x: len(x.split()))

        df_processed['hashtags'] = text_series.apply(self.extract_hashtags)
        df_processed['hashtag_count'] = df_processed['hashtags'].str.len()

        df_processed['mentions'] = text_series.apply(self.extract_mentions)
        df_processed['mention_count'] = df_processed['mentions'].str.len()

        df_processed['urls'] = text_series.apply(self.extract_urls)
        df_processed['url_count'] = df_processed['urls'].str.len()

        df_processed['emoji_count'] = text_series.apply(self.count_emojis)
        df_processed['language'] = text_series.apply(self.detect_language)
        df_processed['is_expo_related'] = text_series.apply(self.is_expo_related)

        logger.info("Text processing complete.")
        return df_processed

    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        if pd.isna(text):
            return []
        return [tag.lower() for tag in self._hashtag_pattern.findall(text)]

    def extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text."""
        if pd.isna(text):
            return []
        return [mention.lower() for mention in self._mention_pattern.findall(text)]

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        if pd.isna(text):
            return []
        return self._url_pattern.findall(text)

    def count_emojis(self, text: str) -> int:
        """Count emojis in text."""
        if pd.isna(text):
            return 0
        return len(self._emoji_pattern.findall(text))

    def detect_language(self, text: str) -> str:
        """Detect primary language (English, Arabic, mixed, or unknown)."""
        if pd.isna(text):
            return 'unknown'

        has_arabic = bool(self._arabic_pattern.search(text))
        has_latin = bool(self._latin_pattern.search(text))

        if has_arabic and has_latin:
            return 'mixed'
        elif has_arabic:
            return 'arabic'
        elif has_latin:
            return 'english'
        else:
            return 'unknown'

    def is_expo_related(self, text: str) -> bool:
        """Check if text contains Expo-related keywords."""
        if pd.isna(text):
            return False

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.EXPO_KEYWORDS)

    def get_top_items(self, items_list: pd.Series, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Get top N most common items from a series of lists.

        Args:
            items_list: Series containing lists (e.g., hashtags)
            top_n: Number of top items to return

        Returns:
            List of (item, count) tuples
        """
        all_items = [item for sublist in items_list if isinstance(sublist, list) for item in sublist]
        return Counter(all_items).most_common(top_n)

    def remove_stopwords(self, text: str, language: str = 'english') -> str:
        """
        Remove stopwords from text based on language.

        Args:
            text: Input text
            language: Language of stopwords to remove

        Returns:
            Text with stopwords removed
        """
        if pd.isna(text) or not text.strip():
            return ''

        words = text.lower().split()
        if language in self.stopwords:
            words = [w for w in words if w not in self.stopwords[language]]

        return ' '.join(words)

    def generate_embeddings(self, texts: List[str], model_name: str = 'paraphrase-MiniLM-L6-v2') -> np.ndarray:
        """
        Generate sentence embeddings using Sentence-BERT or a fallback.

        Args:
            texts: A list of text strings to embed.
            model_name: The name of the Sentence-BERT model to use.

        Returns:
            A numpy array of embeddings.
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Generating embeddings using SentenceTransformer model: {model_name}")
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, show_progress_bar=True)
            return embeddings
        except ImportError:
            logger.warning(
                "sentence-transformers library not found. "
                "Falling back to TF-IDF as a mock implementation for embeddings. "
                "Run 'pip install sentence-transformers' for full functionality."
            )
            # Mock implementation using TF-IDF
            tfidf = TfidfVectorizer(max_features=384, stop_words='english')  # 384 is MiniLM's dim
            # Replace None or non-string types with empty strings
            sanitized_texts = [str(text) if text is not None else '' for text in texts]
            embeddings = tfidf.fit_transform(sanitized_texts).toarray()
            return embeddings
