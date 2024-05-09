@echo off && chcp 65001

echo working dir is %cd%
echo downloading requirement aria2 check.
echo=
dir /a:d/b | findstr "aria2" > flag.txt
findstr "aria2" flag.txt >nul
if %errorlevel% ==0 (
    echo aria2 checked.
    echo=
) else (
    echo failed. please downloading aria2 from webpage!
    echo unzip it and put in this directory!
    timeout /T 5
    start https://github.com/aria2/aria2/releases/tag/release-1.36.0
    echo=
    goto end
)

echo envfiles checking start.
echo=

for /f %%x in ('findstr /i /c:"aria2" "flag.txt"') do (set aria2=%%x)&goto endSch
:endSch

set d32=f0D32k.pth
set d40=f0D40k.pth
set d48=f0D48k.pth
set g32=f0G32k.pth
set g40=f0G40k.pth
set g48=f0G48k.pth

set d40v2=f0D40k.pth
set g40v2=f0G40k.pth

set dld32=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth
set dld40=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth
set dld48=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth
set dlg32=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth
set dlg40=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth
set dlg48=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth

set dld40v2=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth
set dlg40v2=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth

set hb=hubert_base.pt

set dlhb=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt

set rmvpe=rmvpe.pt
set dlrmvpe=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt

echo dir check start.
echo=

if exist "%~dp0assets\pretrained" (
        echo dir .\assets\pretrained checked.
    ) else (
        echo failed. generating dir .\assets\pretrained.
        mkdir pretrained
    )
if exist "%~dp0assets\pretrained_v2" (
        echo dir .\assets\pretrained_v2 checked.
    ) else (
        echo failed. generating dir .\assets\pretrained_v2.
        mkdir pretrained_v2
    )    

echo=
echo dir check finished.

echo=
echo required files check start.

echo checking D32k.pth
if exist "%~dp0assets\pretrained\D32k.pth" (
        echo D32k.pth in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth -d %~dp0assets\pretrained -o D32k.pth
        if exist "%~dp0assets\pretrained\D32k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking D40k.pth
if exist "%~dp0assets\pretrained\D40k.pth" (
        echo D40k.pth in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth -d %~dp0assets\pretrained -o D40k.pth
        if exist "%~dp0assets\pretrained\D40k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking D40k.pth
if exist "%~dp0assets\pretrained_v2\D40k.pth" (
        echo D40k.pth in .\assets\pretrained_v2 checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth -d %~dp0assets\pretrained_v2 -o D40k.pth
        if exist "%~dp0assets\pretrained_v2\D40k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )    
echo checking D48k.pth
if exist "%~dp0assets\pretrained\D48k.pth" (
        echo D48k.pth in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D48k.pth -d %~dp0assets\pretrained -o D48k.pth
        if exist "%~dp0assets\pretrained\D48k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking G32k.pth
if exist "%~dp0assets\pretrained\G32k.pth" (
        echo G32k.pth in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G32k.pth -d %~dp0assets\pretrained -o G32k.pth
        if exist "%~dp0assets\pretrained\G32k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking G40k.pth
if exist "%~dp0assets\pretrained\G40k.pth" (
        echo G40k.pth in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G40k.pth -d %~dp0assets\pretrained -o G40k.pth
        if exist "%~dp0assets\pretrained\G40k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking G40k.pth
if exist "%~dp0assets\pretrained_v2\G40k.pth" (
        echo G40k.pth in .\assets\pretrained_v2 checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth -d %~dp0assets\pretrained_v2 -o G40k.pth
        if exist "%~dp0assets\pretrained_v2\G40k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )    
echo checking G48k.pth
if exist "%~dp0assets\pretrained\G48k.pth" (
        echo G48k.pth in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G48k.pth -d %~dp0assets\pretrained -o G48k.pth
        if exist "%~dp0assets\pretrained\G48k.pth" (echo download successful.) else (echo please try again!
        echo=)
    )

echo checking %d32%
if exist "%~dp0assets\pretrained\%d32%" (
        echo %d32% in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dld32% -d %~dp0assets\pretrained -o %d32%
        if exist "%~dp0assets\pretrained\%d32%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %d40%
if exist "%~dp0assets\pretrained\%d40%" (
        echo %d40% in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dld40% -d %~dp0assets\pretrained -o %d40%
        if exist "%~dp0assets\pretrained\%d40%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %d40v2%
if exist "%~dp0assets\pretrained_v2\%d40v2%" (
        echo %d40v2% in .\assets\pretrained_v2 checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dld40v2% -d %~dp0assets\pretrained_v2 -o %d40v2%
        if exist "%~dp0assets\pretrained_v2\%d40v2%" (echo download successful.) else (echo please try again!
        echo=)
    )    
echo checking %d48%
if exist "%~dp0assets\pretrained\%d48%" (
        echo %d48% in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dld48% -d %~dp0assets\pretrained -o %d48%
        if exist "%~dp0assets\pretrained\%d48%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %g32%
if exist "%~dp0assets\pretrained\%g32%" (
        echo %g32% in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlg32% -d %~dp0assets\pretrained -o %g32%
        if exist "%~dp0assets\pretrained\%g32%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %g40%
if exist "%~dp0assets\pretrained\%g40%" (
        echo %g40% in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlg40% -d %~dp0assets\pretrained -o %g40%
        if exist "%~dp0assets\pretrained\%g40%" (echo download successful.) else (echo please try again!
        echo=)
    )
echo checking %g40v2%
if exist "%~dp0assets\pretrained_v2\%g40v2%" (
        echo %g40v2% in .\assets\pretrained_v2 checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlg40v2% -d %~dp0assets\pretrained_v2 -o %g40v2%
        if exist "%~dp0assets\pretrained_v2\%g40v2%" (echo download successful.) else (echo please try again!
        echo=)
    )    
echo checking %g48%
if exist "%~dp0assets\pretrained\%g48%" (
        echo %g48% in .\assets\pretrained checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlg48% -d %~dp0assets\pretrained -o %g48%
        if exist "%~dp0assets\pretrained\%g48%" (echo download successful.) else (echo please try again!
        echo=)
    )

echo checking %hb%
if exist "%~dp0assets\hubert\%hb%" (
        echo %hb% in .\assets\hubert checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlhb% -d %~dp0assets\hubert\ -o %hb%
        if exist "%~dp0assets\hubert\%hb%" (echo download successful.) else (echo please try again!
        echo=)
    )

echo checking %rmvpe%
if exist "%~dp0assets\rmvpe\%rmvpe%" (
        echo %rmvpe% in .\assets\rmvpe checked.
        echo=
    ) else (
        echo failed. starting download from huggingface.
        %~dp0%aria2%\aria2c --console-log-level=error -c -x 16 -s 16 -k 1M %dlrmvpe% -d %~dp0assets\rmvpe\ -o %rmvpe%
        if exist "%~dp0assets\rmvpe\%rmvpe%" (echo download successful.) else (echo please try again!
        echo=)
    )

echo required files check finished.
echo envfiles check complete.
pause
:end
del flag.txt
