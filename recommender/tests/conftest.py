from pathlib import Path

import pytest

from adaptive_recommender import QuestionBank

ROOT = Path(__file__).resolve().parents[1]          # recommender/
REPO = ROOT.parent                                    # repository root (data/ lives there)


@pytest.fixture(scope="session")
def bank():
    return QuestionBank.from_csv(ROOT / "data" / "question_bank.csv")
