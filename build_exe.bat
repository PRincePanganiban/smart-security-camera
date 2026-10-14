@echo off
REM ============================================================
REM  Builds SmartSecurityCamera.exe  (run this on Windows)
REM  Double-click it, or run it from the VS Code terminal:
REM      build_exe.bat
REM ============================================================
cd /d "%~dp0"

if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

echo.
echo [1/4] Installing requirements and PyInstaller...
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

echo.
echo [2/4] Building the .exe (this takes a few minutes)...
python -m PyInstaller --noconfirm --onefile --windowed --name SmartSecurityCamera --add-data "assets;assets" app.py
if errorlevel 1 goto :error

echo.
echo [3/4] Copying sample videos next to the .exe...
xcopy /E /I /Y sample_inputs dist\sample_inputs >nul

echo.
echo [4/4] Running a quick self-test of the .exe...
start /wait "" dist\SmartSecurityCamera.exe --selftest
if exist dist\selftest.txt type dist\selftest.txt

echo.
echo ============================================================
echo  Done!  Your program is:  dist\SmartSecurityCamera.exe
echo  Share the whole "dist" folder (the exe + sample_inputs).
echo ============================================================
pause
exit /b 0

:error
echo.
echo [ERROR] Build failed. Scroll up to read the message above.
pause
exit /b 1
