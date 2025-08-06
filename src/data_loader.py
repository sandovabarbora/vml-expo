"""Data loading functionality for social media analysis."""

from dataclasses import dataclass
from enum import Enum
import json
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported social media platforms."""
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    UNKNOWN = "unknown"

    @classmethod
    def from_url(cls, url: Union[str, float, None]) -> 'Platform':
        """
        Determine platform from URL.

        Args:
            url: URL string or None/NaN

        Returns:
            Platform enum value
        """
        # Handle None, NaN, or empty values
        if pd.isna(url) or not url:
            return cls.UNKNOWN

        # Ensure we have a string
        if not isinstance(url, str):
            return cls.UNKNOWN

        url_lower = url.lower()
        if 'instagram.com' in url_lower:
            return cls.INSTAGRAM
        elif 'twitter.com' in url_lower:
            return cls.TWITTER
        else:
            return cls.UNKNOWN


@dataclass
class DataLoadResult:
    """Result of data loading operation."""
    dataframe: pd.DataFrame
    errors: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SocialMediaDataLoader:
    """Load and validate social media data from JSON files."""

    def __init__(self, filepath: Path):
        """
        Initialize data loader.

        Args:
            filepath: Path to the JSON data file
        """
        self.filepath = Path(filepath)
        self._validate_file_exists()

    def _validate_file_exists(self) -> None:
        """Ensure the data file exists."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

    def load(self) -> DataLoadResult:
        """
        Load data from JSON file with error handling.

        Returns:
            DataLoadResult containing dataframe, errors, and metadata
        """
        records = []
        errors = []

        logger.info(f"Loading data from {self.filepath}")

        with open(self.filepath, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if not line.strip():  # Skip empty lines
                    continue

                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError as e:
                    errors.append({
                        'line': line_num,
                        'error': str(e),
                        'content': line[:100]
                    })

        df = pd.DataFrame(records)

        # Process and validate data
        df = self._process_dataframe(df)

        metadata = {
            'total_lines': line_num,
            'successful_records': len(records),
            'failed_records': len(errors),
            'file_size_mb': self.filepath.stat().st_size / (1024 * 1024)
        }

        logger.info(f"Loaded {len(df)} records with {len(errors)} errors")

        return DataLoadResult(
            dataframe=df,
            errors=errors,
            metadata=metadata
        )

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and validate the loaded dataframe.

        Args:
            df: Raw dataframe

        Returns:
            Processed dataframe
        """
        # Convert timestamp
        if 'created_time' in df.columns:
            df['created_time'] = pd.to_datetime(df['created_time'], errors='coerce')

        # Add derived platform column
        if 'link' in df.columns:
            df['platform'] = df['link'].apply(lambda x: Platform.from_url(x).value)

        # Ensure numeric columns
        if 'interaction_count' in df.columns:
            df['interaction_count'] = pd.to_numeric(df['interaction_count'], errors='coerce')

        # Sort by timestamp for time series analysis
        if 'created_time' in df.columns:
            df = df.sort_values('created_time').reset_index(drop=True)

        return df
