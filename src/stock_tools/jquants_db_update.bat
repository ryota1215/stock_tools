@REM @echo off
@REM @if not "%~0"=="%~dp0.\%~nx0" start /min cmd /c,"%~dp0.\%~nx0" %* & goto :eof

call activate
cd C:\Users\ryota\sys_trading\stock_tools\src\stock_tools
python db_maker.py
python drive_uploader.py
python labo_future.py
pause