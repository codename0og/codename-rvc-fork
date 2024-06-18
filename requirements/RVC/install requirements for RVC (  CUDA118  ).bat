@echo off
REM Change to the "reclists-RVC" directory if it exists
if exist "reclists-RVC" (
    cd /d "reclists-RVC"
)

REM Install torch, torchvision, and torchaudio  all with cuda-ver-118 support
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

REM Install packages listed in the requirements file for rvc ( cuda 118 )
pip install -r rvc-requirements.txt

REM Add error handling if needed
if %errorlevel% neq 0 (
    echo An error occurred during installation.
    exit /b %errorlevel%
)