@REM @echo off
@REM @if not "%~0"=="%~dp0.\%~nx0" start /min cmd /c,"%~dp0.\%~nx0" %* & goto :eof

call activate
python db_maker.py
pause