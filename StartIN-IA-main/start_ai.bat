@echo off
title StartIn AI Service
echo ========================================================
echo       Iniciando Servico de IA Python - StartIn
echo ========================================================
cd /d %~dp0
set PYTHONIOENCODING=utf-8
python server.py
pause
