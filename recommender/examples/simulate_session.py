"""Play one adaptive session with a simulated learner and print what the recommender did.

    python examples/simulate_session.py                              # hill climbing (MOO), ability 0.6
    python examples/simulate_session.py --method mab --policy thompson --ability 0.4 --seed 3
"""
import argparse
from pathlib import Path

from adaptive_recommender import LearningSession, QuestionBank, SimulatedLearner, run_session, toft

BANK = Path(__file__).resolve().parent.parent / "data" / "question_bank.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["hillclimbing", "mab"], default="hillclimbing")
    p.add_argument("--variant", default="MOO", choices=["MOO", "MOEG", "MOAG", "MOAE"], help="hill-climbing variant (method=hillclimbing)")
    p.add_argument("--policy", default="thompson", choices=["thompson", "egreedy", "softmax", "ucb", "random"], help="bandit policy (method=mab)")
    p.add_argument("--ability", type=float, default=0.6, help="latent ability of the simulated learner, in [0, 1]")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bank = QuestionBank.from_csv(BANK)
    session = LearningSession(bank, method=args.method, variant=args.variant, policy=args.policy, seed=args.seed)
    run_session(session, SimulatedLearner(ability=args.ability, seed=1000 + args.seed))

    pre = [a for a in session.answers if a.phase == "pretest"]
    print(f"pretest: {sum(a.correct for a in pre)}/{len(pre)} correct, mastery {pre[-1].mastery_after:.2f}\n")
    print(f"{'batch':>5} {'by':>5}  {'difficulties of the questions (✓ correct / ✗ failed)':<48} {'mastery':>7} {'AUFS':>6} {'AEFS':>6}")
    for b in session.batches:
        answers = [a for a in session.answers if a.phase == "learning" and a.batch_index == b.batch_index]
        questions = "  ".join(f"{a.question.difficulty:.2f}{'✓' if a.correct else '✗'}" for a in answers)
        print(f"{b.batch_index + 1:>5} {b.variant:>5}  {questions:<48} {b.mastery_after:>7.2f} {b.aufs_after:>6.2f} {b.aefs_after:>6.2f}")
    aufs = [b.aufs_after for b in session.batches]
    print(f"\nfinal mastery {session.mastery:.2f} after {len(session.batches)} batches; TOFT (AUFS > 0.5) = {toft(aufs):.2f}")


if __name__ == "__main__":
    main()
