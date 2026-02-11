@echo off
REM Quick install script for all DevMaster tools (Windows)

echo 🚀 Installing all DevMaster tools...
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

cd /d "%SCRIPT_DIR%"

REM Install core tools (fast)
echo 📦 Installing AI Debug Companion...
cd ai-debug-companion
pip install -e .
cd ..
echo.

echo 📦 Installing DevNarrative...
cd devnarrative
pip install -e .
cd ..
echo.

echo 📦 Installing CodeArchaeology...
cd codearchaeology
pip install -e .
cd ..
echo.

echo 📦 Installing DevMaster (CLI)...
cd devmaster
pip install -e .
cd ..
echo.

echo ✅ Core tools installed!
echo.
echo ⚠️  CodeSeek and DevKnowledge have heavy ML dependencies
echo     They're optional for the demo. Install later if needed.
echo.
echo 🎯 Test it: devmaster status
echo.

pause
