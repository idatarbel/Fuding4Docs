@echo off
:loop
cls
nvidia-smi
timeout /t 1 /nobreak >nul
goto loop