#!/bin/bash

# 3DRC Pipeline Execution Wrapper Script

# 1. Initialize Conda (to enable conda commands regardless of the shell environment)
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

# 2. Activate the 3drc virtual environment
conda activate 3drc

# 3. Execute the Python script (passing all arguments)
python src/sfm_pipeline.py "$@"
