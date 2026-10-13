@echo off
set "CONDA_ROOT=%USERPROFILE%\miniforge3"

call "%CONDA_ROOT%\Scripts\activate.bat" s3d
if errorlevel 1 (
    echo Could not activate the s3d conda environment.
    pause
    exit /b 1
)

python "%~dp0suite3d\viewer\info_viewer_gui.py" %*
