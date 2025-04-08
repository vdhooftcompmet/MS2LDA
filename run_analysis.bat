@echo off
rem Set the console title (optional)
title Run MS2LDA CLI

rem Set PYTHONPATH to include the current directory (%CD%)
rem This allows Python to find the MS2LDA package when the script in App\ is run
set PYTHONPATH=%CD%;%PYTHONPATH%

echo PYTHONPATH set to: %PYTHONPATH%
echo Running MS2LDA CLI script...
echo Command: python scripts\ms2lda_runfull.py %*
echo.

rem Execute the Python script located in App\, passing all arguments (%*) received by this batch script
python scripts\ms2lda_runfull.py %*

echo.
echo CLI script finished.
