import os
import sys
import tempfile
from pathlib import Path

WEBAPP = Path(__file__).resolve().parents[1]

# The application reads its configuration from the environment at import time: use a throw-away database.
_tmp = tempfile.mkdtemp(prefix="webapp-tests-")
os.environ["STUDY_DB_PATH"] = str(Path(_tmp) / "study.sqlite")
os.environ["STUDY_ALLOW_DEMO"] = "1"
os.environ.pop("STUDY_DB_URL", None)
os.environ.pop("STUDY_QUESTIONS_PATH", None)
sys.path.insert(0, str(WEBAPP))
