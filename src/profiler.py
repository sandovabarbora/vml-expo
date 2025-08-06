"""Data profiling functionality."""

from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd


@dataclass
class ColumnProfile:
    """Profile information for a single column."""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float


class DataProfiler:
    """Generate comprehensive data profile."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def profile(self) -> Dict[str, Any]:
        """Generate complete data profile."""
        return {
            'shape': self._get_shape_info(),
            'memory': self._get_memory_info(),
            'columns': self._profile_columns(),
            'temporal': self._profile_temporal(),
            'quality': self._assess_quality()
        }

    def _get_shape_info(self) -> Dict[str, int]:
        """Get dataframe shape information."""
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'duplicates': self.df.duplicated().sum()
        }

    def _get_memory_info(self) -> Dict[str, float]:
        """Get memory usage information."""
        memory_bytes = self.df.memory_usage(deep=True).sum()
        return {
            'bytes': memory_bytes,
            'mb': memory_bytes / (1024 ** 2)
        }

    def _profile_columns(self) -> List[ColumnProfile]:
        """Profile each column."""
        profiles = []

        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            unique_count = self.df[col].nunique()

            profile = ColumnProfile(
                name=col,
                dtype=str(self.df[col].dtype),
                null_count=null_count,
                null_percentage=null_count / len(self.df) * 100,
                unique_count=unique_count,
                unique_percentage=unique_count / len(self.df) * 100
            )
            profiles.append(profile)

        return profiles

    def _profile_temporal(self) -> Dict[str, Any]:
        """Profile temporal aspects of data."""
        if 'created_time' not in self.df.columns:
            return {}

        return {
            'min_date': self.df['created_time'].min(),
            'max_date': self.df['created_time'].max(),
            'date_range_days': (self.df['created_time'].max() - self.df['created_time'].min()).days,
            'posts_per_day': len(self.df) / ((self.df['created_time'].max() - self.df['created_time'].min()).days + 1)
        }

    def _assess_quality(self) -> Dict[str, float]:
        """Assess overall data quality."""
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()

        return {
            'completeness': (total_cells - missing_cells) / total_cells * 100,
            'has_duplicates': self.df.duplicated().any()
        }
