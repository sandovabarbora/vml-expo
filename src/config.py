from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "dataset_expo2020.json"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
INSIGHTS_DIR = OUTPUT_DIR / "insights"

# Create directories if they don't exist
for directory in [FIGURES_DIR, MODELS_DIR, INSIGHTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Expo 2020 metadata
EXPO_START_DATE = "2021-10-01"
EXPO_END_DATE = "2022-03-31"
EXPO_THEMES = ["Opportunity", "Mobility", "Sustainability"]