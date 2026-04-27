@echo off
cd /d C:\Users\tonysong\Desktop\AI_Python\mcp-server-template
set DEEPSEEK_API_KEY=sk-47196ba610fb4e8eafaabafb331f7f37
.venv\Scripts\python.exe src/DB/server.py %*
