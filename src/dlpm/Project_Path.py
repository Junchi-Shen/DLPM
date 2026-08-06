"""Repository-local paths for the reproducible research release.

Large market data, checkpoints, and path archives are deliberately excluded
from version control.  Set the optional environment variables below to point
to locally stored artifacts; otherwise the repository-local directories are
used.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DLPM_DATA_ROOT", PROJECT_ROOT / "data" / "processed")).resolve()
ARTIFACT_ROOT = Path(os.environ.get("DLPM_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")).resolve()

Pipelines_DIR = PROJECT_ROOT / "scripts"
DATA_CODE_DIR = PROJECT_ROOT / "src" / "dlpm" / "Data"
Trainning_DATA_DIR = DATA_ROOT / "Trainning_Dataset"
Testing_DATA_DIR = DATA_ROOT / "Testing_Dataset"
Validation_DATA_DIR = DATA_ROOT / "Validation_Dataset"
Model_DIR = PROJECT_ROOT / "src" / "dlpm" / "Model"
Unet_Model_DIR = Model_DIR / "Diffusion_Model"
Garch_Model_DIR = ARTIFACT_ROOT / "rnq_models"
Config_DIR = PROJECT_ROOT / "src" / "dlpm" / "Config"
Results_DIR = ARTIFACT_ROOT / "results"
Model_Results_DIR = ARTIFACT_ROOT / "checkpoints"
Path_Generator_Results_DIR = ARTIFACT_ROOT / "path_archives"
Report_Results_DIR = ARTIFACT_ROOT / "reports"
Game_DIR = PROJECT_ROOT / "src" / "dlpm" / "Game"
