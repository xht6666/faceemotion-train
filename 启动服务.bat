@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   FER+ 6 分类情绪识别服务
echo   页面:  http://127.0.0.1:8000/
echo   文档:  http://127.0.0.1:8000/docs
echo   按 Ctrl+C 停止
echo ============================================
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
