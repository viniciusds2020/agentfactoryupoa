@echo off
setlocal

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\clear_vector_db.py --yes
) else (
  python scripts\clear_vector_db.py --yes
)

endlocal
