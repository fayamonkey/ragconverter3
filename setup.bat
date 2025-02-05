@echo off
setlocal enabledelayedexpansion

echo RAG Converter Setup
echo Copyright (c) 2024 Ai-engineering.ai - Dirk Wonhöfer
echo.

:: Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Starting download and installation...
    
    :: Download Python installer
    curl -o python_installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    
    :: Install Python
    echo Installing Python...
    python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    
    :: Clean up
    del python_installer.exe
    
    echo Python has been installed.
    echo.
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual Python environment...
    python -m venv venv
    echo.
)

:: Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat

:: Install requirements
echo Installing required packages...
pip install -r requirements.txt
echo.

:: Install spaCy model
echo Installing English language model...
python -m spacy download en_core_web_sm
echo.

:: Create desktop shortcut
echo Creating desktop shortcut...
set SCRIPT_PATH=%~dp0
set DESKTOP_PATH=%USERPROFILE%\Desktop

:: Create a VBS script to make the shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%DESKTOP_PATH%\RAG Converter.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%SCRIPT_PATH%run.bat" >> CreateShortcut.vbs
echo oLink.IconLocation = "%SCRIPT_PATH%icon.ico" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%SCRIPT_PATH%" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs

:: Run the VBS script and then delete it
cscript //nologo CreateShortcut.vbs
del CreateShortcut.vbs

:: Create run.bat
echo @echo off > run.bat
echo call venv\Scripts\activate.bat >> run.bat
echo start http://localhost:8501 >> run.bat
echo streamlit run src/main.py >> run.bat

:: Download icon if it doesn't exist
if not exist "icon.ico" (
    echo Downloading app icon...
    curl -o icon.ico https://raw.githubusercontent.com/streamlit/streamlit/develop/app/static/favicon.ico
)

echo.
echo Installation completed!
echo.
echo You can now start the RAG Converter using the desktop shortcut.
echo.
pause 