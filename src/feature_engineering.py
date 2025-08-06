# src/feature_engineering.py
"""Feature engineering based on validated hypotheses from statistical testing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering based on hypothesis testing results."""

    # Constants validated through statistical testing
    EMOJI_OPTIMAL_RANGE = (2, 3)  # H3: Confirmed optimal range
    EMOJI_EXCESSIVE_THRESHOLD = 5  # H3: Performance drops after this
    HASHTAG_OPTIMAL_RANGE = (10, 15)  # H8: Peak performance range
    HASHTAG_MAX = 30  # H8: Cap for outliers
    OUTLIER_PERCENTILE = 0.999
    VIRAL_HASHTAGS = ['#expo2020', '#dubai', '#uae', '#expo2020dubai']

    # Content types ranked by performance (from post-hoc tests)
    HIGH_ENGAGEMENT_TYPES = ['carousel', 'photo', 'image']
    LOW_ENGAGEMENT_TYPES = ['status', 'link']

    def __init__(self, expo_start_date: str = "2021-10-01"):
        self.expo_start = pd.Timestamp(expo_start_date, tz='UTC')
        self.fitted = False
        self.outlier_threshold = None
        self.platform_means = {}

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit on training data to learn thresholds."""
        self.outlier_threshold = df['interaction_count'].quantile(self.OUTLIER_PERCENTILE)

        # Learn platform-specific means for scaling
        self.platform_means = df.groupby('platform')['interaction_count'].mean().to_dict()

        self.fitted = True
        logger.info(f"Fitted on {len(df)} samples. Outlier threshold: {self.outlier_threshold:.0f}")
        logger.info(f"Platform means: Instagram={self.platform_means.get('instagram', 0):.1f}, "
                    f"Twitter={self.platform_means.get('twitter', 0):.1f}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe with engineered features."""
        if not self.fitted:
            raise ValueError("Must fit before transform")

        df = df.copy()

        # Apply features in order of importance (based on effect sizes)
        df = self._add_platform_features(df)  # H2: rank-biserial = -0.41
        df = self._add_content_type_features(df)  # H1: ε² = 0.18
        df = self._add_hashtag_features(df)  # H8: ρ = 0.31 (non-linear)
        df = self._add_emoji_features(df)  # H3: Confirmed non-linear
        df = self._add_url_features(df)  # H4: Platform-specific paradox
        df = self._add_interaction_features(df)  # Cross-feature interactions
        df = self._add_target_features(df)

        # Drop temporal features (H5 showed minimal effect)
        # Only keep minimal temporal for control
        df['hour'] = df['created_time'].dt.hour
        df['is_weekend'] = df['created_time'].dt.dayofweek.isin([5, 6]).astype(int)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _add_platform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """H2: Platform has strong effect (rank-biserial = -0.41)."""
        # Platform is already in df, but add platform-specific scaling factor
        df['platform_weight'] = df['platform'].map(
            lambda x: self.platform_means.get(x, 1) / max(self.platform_means.values())
        )
        return df

    def _add_content_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """H1: Content type strongly affects engagement (ε² = 0.18)."""
        # Binary indicators for high/low performing types
        df['is_visual_content'] = df['type_filled'].isin(self.HIGH_ENGAGEMENT_TYPES).astype(int)
        df['is_text_content'] = df['type_filled'].isin(self.LOW_ENGAGEMENT_TYPES).astype(int)

        # Content type ranking based on post-hoc tests
        content_ranks = {
            'carousel': 4, 'photo': 3, 'image': 3,
            'video': 2, 'link': 1, 'status': 0
        }
        df['content_rank'] = df['type_filled'].map(content_ranks).fillna(1)

        return df

    def _add_hashtag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """H8: Non-linear monotonic relationship (Spearman=0.31, Pearson=-0.002)."""
        # Cap hashtags
        df['hashtag_capped'] = df['hashtag_count'].clip(upper=self.HASHTAG_MAX)

        # Non-linear transformation to capture rank correlation
        df['hashtag_log'] = np.log1p(df['hashtag_count'])
        df['hashtag_sqrt'] = np.sqrt(df['hashtag_count'])

        # Optimal range indicator
        df['hashtag_optimal'] = df['hashtag_count'].between(*self.HASHTAG_OPTIMAL_RANGE).astype(int)

        # Viral hashtag presence
        df['has_viral_hashtag'] = df['hashtags'].apply(
            lambda x: int(any(tag in self.VIRAL_HASHTAGS for tag in x))
        )

        # Hashtag saturation (diminishing returns)
        df['hashtag_saturated'] = (df['hashtag_count'] > 20).astype(int)

        return df

    def _add_emoji_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """H3: Confirmed non-linear relationship with optimal at 2-3."""
        # Optimal range indicator (strongest signal)
        df['emoji_optimal'] = df['emoji_count'].between(*self.EMOJI_OPTIMAL_RANGE).astype(int)

        # Problem indicators
        df['emoji_none'] = (df['emoji_count'] == 0).astype(int)
        df['emoji_excessive'] = (df['emoji_count'] > self.EMOJI_EXCESSIVE_THRESHOLD).astype(int)

        # Non-linear encoding
        df['emoji_squared'] = df['emoji_count'] ** 2

        return df

    def _add_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """H4: Simpson's paradox - negative overall but positive per platform."""
        # ONLY platform-specific URL features (no general has_url)
        df['instagram_url'] = (
                (df['platform'] == 'instagram') & (df['url_count'] > 0)
        ).astype(int)

        df['twitter_url'] = (
                (df['platform'] == 'twitter') & (df['url_count'] > 0)
        ).astype(int)

        # URL * platform interaction captures the paradox
        df['url_platform_interaction'] = df['url_count'] * (
            df['platform'].map({'instagram': 2.1, 'twitter': 3.7}).fillna(1)
        )

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interaction features based on hypothesis testing insights."""
        # Platform * Content type (both have large effects)
        df['platform_visual'] = (
                (df['platform'] == 'instagram') & df['is_visual_content']
        ).astype(int)

        # Hashtag * Platform (different optimal ranges)
        df['hashtag_platform'] = df['hashtag_capped'] * df['platform_weight']

        # Content density
        df['content_density'] = (
                                        df['hashtag_log'] + df['mention_count'] +
                                        df['url_count'] + df['emoji_optimal']
                                ) / df['word_count'].clip(lower=1)

        return df

    def _add_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Target transformations."""
        df['has_engagement'] = (df['interaction_count'] > 0).astype(int)
        df['log_engagement'] = np.log1p(df['interaction_count'])
        df['is_outlier'] = (df['interaction_count'] > self.outlier_threshold).astype(int)

        # Engagement bins based on distribution
        df['engagement_bin'] = pd.cut(
            df['interaction_count'],
            bins=[0, 1, 10, 100, 1000, float('inf')],
            labels=['zero', 'low', 'medium', 'high', 'viral'],
            include_lowest=True
        )

        return df

    def get_feature_columns(self) -> Dict[str, List[str]]:
        """Get organized feature lists by importance."""
        return {
            'numerical': [
                # Platform features (H2: highest effect)
                'platform_weight',

                # Content features (H1: second highest)
                'is_visual_content', 'is_text_content', 'content_rank',

                # Hashtag features (H8: ρ=0.31)
                'hashtag_capped', 'hashtag_log', 'hashtag_sqrt',
                'hashtag_optimal', 'hashtag_saturated', 'has_viral_hashtag',

                # Emoji features (H3: confirmed non-linear)
                'emoji_optimal', 'emoji_none', 'emoji_excessive', 'emoji_squared',

                # URL features (H4: platform-specific only)
                'instagram_url', 'twitter_url', 'url_platform_interaction',

                # Interaction features
                'platform_visual', 'hashtag_platform', 'content_density',

                # Basic features
                'text_length', 'word_count', 'mention_count',

                # Minimal temporal (low importance)
                'hour', 'is_weekend',

                # Target indicators
                'is_outlier'
            ],
            'categorical': [
                'platform',  # Most important
                'type_filled',  # Second most important
                'language',
                'engagement_bin'  # For stratified sampling
            ]
        }