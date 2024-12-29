#!/bin/bash

# Create a virtual environment
python -m venv new_venv

# Activate the virtual environment
source new_venv/bin/activate

# Install PyTorch (CPU-only)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric dependencies
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric
pip install numpy<2

# Install all remaining dependencies
pip install -r requirements.txt

# Completion message
echo "Installation complete. Activate the environment with:"
echo "source new_venv/bin/activate"
