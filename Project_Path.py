# Project_Path.py
# This module defines the project root and data directory paths.
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.resolve()

# Define directories for pipelines
Pipelines_DIR = PROJECT_ROOT / "Pipelines" #√

# Define the data directory relative to the project root
DATA_CODE_DIR = PROJECT_ROOT / "Data" #√

# Define specific dataset directories
Trainning_DATA_DIR = PROJECT_ROOT/ "Dataset" / "Trainning_Dataset"#√
Testing_DATA_DIR = PROJECT_ROOT / "Dataset" / "Testing_Dataset"#√



# Define directories for different components of the project
Explainer_DIR = PROJECT_ROOT / "Explainer"#√
Generator_DIR = PROJECT_ROOT/ "Generator"#√

# Define directories for specific models
Model_DIR = PROJECT_ROOT / "Model"#√
Unet_Model_DIR = Model_DIR / "Diffusion_Model"#√
Garch_Model_DIR = Model_DIR/ "Garch_Model"#√

# Define directories for option pricing
Game_DIR = PROJECT_ROOT / "Game"#√

# Define directories for results
Results_DIR = PROJECT_ROOT / "Results"
Model_Results_DIR = Results_DIR / "Model_Results"#√
Path_Generator_Results_DIR = Results_DIR / "Path_Generator_Results"#√
Report_Results_DIR = Results_DIR / "Report_Results"#√


#
Config_DIR = PROJECT_ROOT / "Config"



# Function to create all necessary directories
def create_project_directories(verbose=True):
    """
    Ensures all necessary project directories exist.
    If a directory does not exist, it will be created.

    Args:
        verbose (bool): If True, prints a message for each directory created.
                        If False, runs silently.
    """
    dirs_to_create = [

        DATA_CODE_DIR,
        Trainning_DATA_DIR,
        Testing_DATA_DIR,
        Explainer_DIR,
        Generator_DIR,
        Model_DIR,
        Unet_Model_DIR,
        Garch_Model_DIR,
        Game_DIR,
        Results_DIR, 
        Model_Results_DIR,
        Path_Generator_Results_DIR,
        Report_Results_DIR,
        Pipelines_DIR,
        Config_DIR,
]

    print("--- Ensuring project directories exist ---")
    for _dir in dirs_to_create:
        # Check if directory exists before attempting creation and printing
        if not _dir.exists():
            _dir.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Created directory: {_dir}")
        elif verbose:
            print(f"Directory already exists: {_dir}")
    print("--- Directory setup complete ---")


#create_project_directories()