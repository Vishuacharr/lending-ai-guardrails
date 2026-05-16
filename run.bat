@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Running tests...
python -m pytest tests/ -v

echo.
echo Launching dashboard...
streamlit run app.py
