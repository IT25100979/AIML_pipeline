@echo off
cd /d "%~dp0"
echo Starting the Ames Housing local project website...
echo The local server will run in a minimized command window.
start "Ames Local Website Server" /min cmd /c python -m http.server 8000 --bind 127.0.0.1
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000/website/"
echo Website opened in your browser.
echo Close the minimized server window to stop the website.
