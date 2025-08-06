"""Data validation functionality."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import pandas as pd


class ValidationRule(ABC):
    """Abstract base class for validation rules."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate dataframe against rule.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class RequiredColumnsRule(ValidationRule):
    """Ensure required columns are present."""

    def __init__(self, required_columns: List[str]):
        self.required_columns = required_columns

    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            return False, f"Missing required columns: {missing}"
        return True, ""


class NoFutureDatesRule(ValidationRule):
    """Ensure no dates are in the future."""

    def __init__(self, date_column: str):
        self.date_column = date_column

    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if self.date_column not in df.columns:
            return True, ""

        # Get current time with proper timezone handling
        now = pd.Timestamp.now('UTC')

        # Check if column has timezone info
        if df[self.date_column].dt.tz is None:
            # If timezone-naive, use local now
            now = pd.Timestamp.now()

        future_dates = df[df[self.date_column] > now]
        if len(future_dates) > 0:
            return False, f"Found {len(future_dates)} posts with future dates"
        return True, ""


class DataValidator:
    """Validate social media data against business rules."""

    def __init__(self):
        self.rules: List[ValidationRule] = []

    def add_rule(self, rule: ValidationRule) -> 'DataValidator':
        """Add validation rule."""
        self.rules.append(rule)
        return self

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation rules.

        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        for rule in self.rules:
            is_valid, message = rule.validate(df)
            if not is_valid:
                results['is_valid'] = False
                results['errors'].append({
                    'rule': rule.__class__.__name__,
                    'message': message
                })

        return results
