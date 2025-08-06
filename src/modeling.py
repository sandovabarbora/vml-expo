# src/modeling.py
"""Modeling functionality for social media engagement prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import xgboost as xgb
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Container for model evaluation results."""
    model_name: str
    platform: str
    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    feature_importances: Optional[pd.DataFrame] = None
    predictions: Optional[np.ndarray] = None
    # Enhancement: Add fields for confidence intervals
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    confidence_level: float = 0.95

    def __str__(self):
        return (f"{self.model_name} on {self.platform}: "
                f"AUC={self.auc_roc:.3f}, Precision={self.precision:.3f}, "
                f"Recall={self.recall:.3f}")


class PlatformAwareVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    A voting ensemble that can use different weights for different platforms.
    For this assignment, we implement a standard soft voting classifier
    that can be trained on data from a specific platform.
    """

    def __init__(self, estimators: List[Tuple[str, Any]], voting: str = 'soft', weights: Optional[List[float]] = None):
        """
        Initialize the ensemble.

        Args:
            estimators: List of (name, estimator) tuples.
            voting: 'soft' for predicted probabilities, 'hard' for majority rule.
            weights: List of weights for estimators.
        """
        self.estimators = estimators
        self.named_estimators_ = {name: est for name, est in estimators}
        self.voting = voting
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PlatformAwareVotingEnsemble':
        """
        Fit the base estimators.

        Args:
            X: Training data.
            y: Target values.

        Returns:
            The fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.classes_, _ = np.unique(y, return_inverse=True)

        for name, estimator in self.estimators:
            logger.info(f"Fitting estimator: {name}")
            estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.

        Args:
            X: The input samples.

        Returns:
            The predicted class labels.
        """
        check_is_fitted(self)
        avg = self._predict_proba(X)
        maj = np.argmax(avg, axis=1)
        return self.classes_[maj]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute probabilities of classes for X.

        Args:
            X: The input samples.

        Returns:
            The predicted probabilities.
        """
        check_is_fitted(self)
        return self._predict_proba(X)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Helper for probability prediction."""
        if self.voting == 'soft':
            probas = np.asarray([est.predict_proba(X) for _, est in self.estimators])
            avg = np.average(probas, axis=0, weights=self.weights)
            return avg
        else: # hard voting
            predictions = np.asarray([est.predict(X) for _, est in self.estimators]).T
            # Apply weights by repeating columns
            if self.weights is not None:
                weighted_preds = []
                for i, w in enumerate(self.weights):
                    weighted_preds.extend([predictions[:, i]] * int(w * 10)) # simple weighting
                predictions = np.asarray(weighted_preds).T

            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)),
                axis=1,
                arr=predictions
            )
            # Convert to probabilities
            proba = np.zeros((len(X), len(self.classes_)))
            proba[np.arange(len(X)), maj] = 1
            return proba


class EngagementPredictor:
    """Main class for engagement prediction modeling."""

    def __init__(self, random_state: int = 42):
        """
        Initialize predictor.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Dict[str, Any]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.results: Dict[str, List[ModelResults]] = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        platform: str,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split and scale data for a specific platform.

        Args:
            df: Dataframe containing features and target.
            features: List of feature column names.
            target: Target column name.
            platform: Platform name ('instagram' or 'twitter').
            test_size: Proportion of the dataset to include in the test split.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        platform_df = df[df['platform'] == platform]
        X = platform_df[features].values
        y = platform_df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.scalers[platform] = scaler
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        platform: str,
        feature_names: List[str]
    ):
        """
        Train multiple models and evaluate their performance.

        Args:
            X_train, y_train: Training data.
            X_test, y_test: Testing data.
            platform: Platform name.
            feature_names: List of feature names for importance.
        """
        models_to_train = {
            'LogisticRegression': LogisticRegression(random_state=self.random_state, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(random_state=self.random_state, class_weight='balanced', n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        }

        self.models[platform] = {}
        self.results[platform] = []

        for name, model in models_to_train.items():
            logger.info(f"Training {name} for {platform}...")
            model.fit(X_train, y_train)
            self.models[platform][name] = model

            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

            importances = None
            if hasattr(model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

            result = ModelResults(
                model_name=name,
                platform=platform,
                auc_roc=auc,
                precision=precision,
                recall=recall,
                f1_score=f1,
                feature_importances=importances,
                predictions=y_pred_proba
            )
            self.results[platform].append(result)
            logger.info(str(result))

    def get_best_model(self, platform: str, metric: str = 'auc_roc') -> Tuple[str, Any]:
        """
        Get the best model for a platform based on a metric.

        Args:
            platform: Platform name.
            metric: Metric to optimize.

        Returns:
            Tuple of (model_name, model).
        """
        if platform not in self.results:
            raise ValueError(f"No results for platform {platform}")

        best_result = max(self.results[platform], key=lambda x: getattr(x, metric))
        best_model = self.models[platform][best_result.model_name]

        return best_result.model_name, best_model

    def get_feature_importance_summary(self, platform: str, top_n: int = 15) -> pd.DataFrame:
        """
        Get aggregated feature importance across models.

        Args:
            platform: Platform name.
            top_n: Number of top features to return.

        Returns:
            DataFrame with aggregated feature importances.
        """
        importance_dfs = []
        for result in self.results.get(platform, []):
            if result.feature_importances is not None:
                imp_df = result.feature_importances.copy()
                imp_df['model'] = result.model_name
                importance_dfs.append(imp_df)

        if not importance_dfs:
            return pd.DataFrame()

        all_importances = pd.concat(importance_dfs)
        avg_importance = all_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
        return avg_importance.head(top_n).to_frame()
