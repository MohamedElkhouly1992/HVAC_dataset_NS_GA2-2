@echo off
cd /d "%~dp0"
python run_observed_pareto.py
if errorlevel 1 exit /b 1
python run_surrogate_nsga2.py
pause
