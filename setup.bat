@echo off
REM Create a virtual environment
python -m venv new_venv

REM Activate the virtual environment
call new_venv\Scripts\activate

REM Install PyTorch (CPU-only)
pip install torch==2.1.0  --index-url https://download.pytorch.org/whl/cpu

REM Install PyTorch Geometric dependencies
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric

REM Install all remaining dependencies
pip install -r requirements.txt

echo Installation complete. Activate the environment with:
echo call new_venv\Scripts\activate
pause
