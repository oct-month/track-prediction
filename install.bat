%~dp0INSTSRV.EXE TrackPrediction %~dp0SRVANY.EXE
@reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\TrackPrediction\Parameters" /v "Application" /t REG_SZ /d "%~dp0run.bat" /f
net start TrackPrediction
@REM echo "��װ�ɹ�"
@pause
