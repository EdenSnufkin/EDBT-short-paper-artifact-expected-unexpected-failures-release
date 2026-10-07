"""Generate data/demo_questions.json: a synthetic arithmetic/algebra bank to try the application.

The questions of the study are not distributed. This bank has the same format (see README.md) with 21 difficulty
levels (0.00, 0.05, ..., 1.00) and 4 questions per level; difficulty only grows with the size of the numbers.

    python data/make_demo_questions.py
"""
import json
import random
from pathlib import Path

LEVELS = [round(0.05 * i, 2) for i in range(21)]
PER_LEVEL = 4


def options_around(rng: random.Random, truth: int) -> tuple[list[str], int]:
    wrong = set()
    while len(wrong) < 3:
        candidate = truth + rng.choice([-1, 1]) * rng.randint(1, max(3, abs(truth) // 8 + 2))
        if candidate != truth:
            wrong.add(candidate)
    values = [truth, *sorted(wrong)]
    rng.shuffle(values)
    return [f"${v}$" for v in values], values.index(truth)


def make(rng: random.Random, level: float, kind: int) -> tuple[str, int, list[str], int]:
    n = int(level * 20)                                   # 0..20
    if kind == 0:
        a, b = rng.randint(2, 4 + n), rng.randint(2, 4 + n)
        text, truth = f"Compute ${a} \\times {b}$.", a * b
    elif kind == 1:
        x, a = rng.randint(1, 3 + n), rng.randint(2, 3 + n // 2)
        b = rng.randint(1, 5 + n)
        text, truth = f"Solve ${a}x + {b} = {a * x + b}$ for $x$.", x
    elif kind == 2:
        a, b = rng.randint(2, 3 + n // 2), rng.randint(1, 5 + n)
        text, truth = f"What is the derivative of $f(x) = {a}x^2 + {b}x$ evaluated at $x = 2$?", 4 * a + b
    else:
        a, b, c, d = (rng.randint(1, 3 + n // 2) for _ in range(4))
        text, truth = f"Compute the determinant of $\\begin{{pmatrix}} {a} & {b} \\\\ {c} & {d} \\end{{pmatrix}}$.", a * d - b * c
    options, correct = options_around(rng, truth)
    return text, truth, options, correct


def main() -> None:
    rng = random.Random(0)
    rows = []
    for level in LEVELS:
        for k in range(PER_LEVEL):
            text, _truth, options, correct = make(rng, level, k)
            rows.append({"id": str(len(rows) + 1), "question": text, "difficulty": level, "options": options, "correct_answer": correct})
    out = Path(__file__).with_name("demo_questions.json")
    out.write_text(json.dumps({"questions": rows}, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
