"""
run.py
------
Convenience entry point for the FraudShield AI Application.
Automatically detects and uses virtual environment Python if present.
"""

import sys
import os
import subprocess

# Auto-detect project virtual environment python
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'Scripts', 'python.exe')

if os.path.exists(VENV_PYTHON) and sys.executable.lower() != os.path.abspath(VENV_PYTHON).lower():
    try:
        import flask
    except ImportError:
        # Global python is missing flask — seamlessly relaunch using venv python
        sys.exit(subprocess.call([VENV_PYTHON, __file__] + sys.argv[1:]))

from app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
