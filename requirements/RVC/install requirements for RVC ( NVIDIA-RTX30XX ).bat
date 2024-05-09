@echo off
REM Change to the "reclists-RVC" directory if it exists
if exist "reclists-RVC" (
    cd /d "reclists-RVC"
)

REM Install torch, torchvision, and torchaudio  cuda-ver-117
pip install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cu117

REM Install packages listed in the requirements file for rvc nvidia-rtx30xx-cu117
pip install -r rvc-requirements.txt

REM Add error handling if needed
if %errorlevel% neq 0 (
    echo An error occurred during installation.
    exit /b %errorlevel%
)