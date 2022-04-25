%~dp0INSTSRV.EXE TrackPrediction %~dp0SRVANY.EXE
@reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\TrackPrediction\Parameters" /v "Application" /t REG_SZ /d "%~dp0run.bat" /f
net start TrackPrediction
@REM echo "安装成功"
@pause
