"""rfm_fe: Dataset-agnostic & model-agnostic RFM feature engineering."""

from .config import RFMConfig
from .engineer import RFMFeatureEngineer, engineer_rfm_features

__all__ = [
    "RFMConfig",
    "RFMFeatureEngineer",
    "engineer_rfm_features",
]

__version__ = "0.1.0"
