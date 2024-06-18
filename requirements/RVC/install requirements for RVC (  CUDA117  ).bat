@echo off
REM Change to the "reclists-RVC" directory if it exists
if exist "reclists-RVC" (
    cd /d "reclists-RVC"
)

REM Install torch, torchvision, and torchaudio  all with cuda-ver-117 support
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html

REM Install packages listed in the requirements file for rvc ( cuda 117 )
pip install -r rvc-requirements.txt

REM Add error handling if needed
if %errorlevel% neq 0 (
    echo An error occurred during installation.
    exit /b %errorlevel%
)