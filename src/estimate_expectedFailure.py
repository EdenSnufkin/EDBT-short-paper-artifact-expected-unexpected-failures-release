from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark import DEFAULT_DATASET_PATH, load_dataset
from expectedFailure_models import DifficultyBased, LearningPotentialBased


expectedFailure_MODELS = {
    "difficulty": DifficultyBased,
    "learningpotential": LearningPotentialBased,
}


def _as_trajectories(data: Any) -> tuple[list[list[dict[str, Any]]], bool]:
    if not data:
        return [], False
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data, True
    if isinstance(data, list):
        return [data], False
    raise ValueError("Trajectory JSON must contain a trajectory or a list of trajectories.")


def _question_result_lookup(question_results: Any) -> dict[int, bool]:
    if not isinstance(question_results, dict):
        raise ValueError("Each step action must contain a question_results dict.")
    return {int(question_id): bool(correct) for question_id, correct in question_results.items()}


def _selected_questions(step: dict[str, Any]) -> list[int]:
    selected = step.get("action", {}).get("selected_questions")
    if not isinstance(selected, list):
        raise ValueError("Each step action must contain a selected_questions list.")
    return [int(question_id) for question_id in selected]


def _skill_for_question(question_info: dict[str, Any]) -> str:
    skills = list(question_info["skills"])
    if not skills:
        raise ValueError("Question has no associated skill.")
    return str(skills[0])


def _initial_mastery(
    trajectory: list[dict[str, Any]],
    default_mastery: float,
) -> dict[str, float]:
    first_state = trajectory[0].get("state", {}) if trajectory else {}
    mastery_after = first_state.get("mastery_after")
    if isinstance(mastery_after, dict):
        return {str(skill): float(default_mastery) for skill in mastery_after}
    return {}


def recalculate_trajectory_expectedFailure(
    trajectory: list[dict[str, Any]],
    question_skill_map: dict[int, dict[str, Any]],
    model_name: str = "learningpotential",
    lam: float = 2.0,
    beta: float = 0.1,
    initial_mastery: float = 0.4,
    ncc_window: int = 2,
    keep_initial_step_frozen: bool = True,
) -> list[dict[str, Any]]:
    """Return step, student, and recalculated expectedFailure data for a trajectory.

    Mirrors recalculate_trajectory_unexpectedFailure (same NCC mastery
    reconstruction, same wrong-answer conditioning), but scores each wrong
    answer by how far its difficulty sits *above* mastery instead of below it
    -- an "expected" failure is a wrong answer on a question that was already
    harder than the learner's mastery.
    """
    model_cls = expectedFailure_MODELS[model_name]
    expectedFailure_model = model_cls(expectedFailure=0.0, beta=beta, lam=lam)
    mastery = _initial_mastery(trajectory, default_mastery=initial_mastery)
    ncc_cache: dict[str, dict[float, list[bool]]] = {
        skill: {} for skill in mastery
    }

    expectedFailure_trajectory: list[dict[str, Any]] = []

    for step_index, step in enumerate(trajectory):
        selected_questions = _selected_questions(step)
        question_results = _question_result_lookup(step.get("action", {}).get("question_results"))
        step_expectedFailure: list[float] = []
        step_expectedFailure_instant: list[float] = []
        frozen_initial_step = keep_initial_step_frozen and step_index == 0

        for question_id in selected_questions:
            if question_id not in question_results:
                raise ValueError(f"Missing result for selected question {question_id}.")
            question_info = question_skill_map[question_id]
            skill = _skill_for_question(question_info)
            mastery.setdefault(skill, float(initial_mastery))
            ncc_cache.setdefault(skill, {})

            difficulty = float(question_info["difficulty"])
            correct = question_results[question_id]
            difficulties = [] if correct else [difficulty]

            expectedFailure, expectedFailure_instant = expectedFailure_model.calculate_expectedFailure(
                difficulties,
                mastery[skill],
            )
            step_expectedFailure.append(float(expectedFailure))
            step_expectedFailure_instant.append(float(expectedFailure_instant))

            if frozen_initial_step:
                continue

            for updated_skill in question_info["skills"]:
                updated_skill = str(updated_skill)
                mastery.setdefault(updated_skill, float(initial_mastery))
                skill_cache = ncc_cache.setdefault(updated_skill, {})
                difficulty_cache = skill_cache.setdefault(difficulty, [])
                difficulty_cache.append(correct)
                del difficulty_cache[:-ncc_window]

                if correct and len(difficulty_cache) >= ncc_window and all(difficulty_cache):
                    mastery[updated_skill] = float(max(mastery[updated_skill], difficulty))
                    continue

                if not correct:
                    for difficulty_level in list(skill_cache):
                        if difficulty_level >= difficulty:
                            skill_cache[difficulty_level] = []

            expectedFailure_model._update_expectedFailure(expectedFailure, mastery[skill])

        expectedFailure_trajectory.append(
            {
                "step": step.get("step", step_index),
                "state": {
                    "student_id": step.get("state", {}).get("student_id"),
                    "expectedFailure": [0.0] if frozen_initial_step else step_expectedFailure,
                    "expectedFailure_instant": step_expectedFailure_instant,
                },
            }
        )

    return expectedFailure_trajectory


def recalculate_file(
    input_path: Path,
    output_path: Path,
    dataset_path: Path,
    topic: str | None,
    model_name: str,
    lam: float,
    beta: float,
    initial_mastery: float,
    ncc_window: int,
    keep_initial_step_frozen: bool,
) -> None:
    dataset = load_dataset(dataset_path=dataset_path)
    dataset.resolve_topic(topic)
    with input_path.open(encoding="utf-8") as file:
        data = json.load(file)

    trajectories, was_nested = _as_trajectories(data)
    updated = [
        recalculate_trajectory_expectedFailure(
            trajectory=trajectory,
            question_skill_map=dataset.question_skill_map,
            model_name=model_name,
            lam=lam,
            beta=beta,
            initial_mastery=initial_mastery,
            ncc_window=ncc_window,
            keep_initial_step_frozen=keep_initial_step_frozen,
        )
        for trajectory in trajectories
    ]

    output_data = updated if was_nested else (updated[0] if updated else [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalculate and save expectedFailure values for existing trajectories."
    )
    parser.add_argument("input", type=Path, help="Input dataset/topic trajectories JSON file.")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Preprocessed dataset JSON (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--topic",
        help="Topic used by the trajectories. Inferred for a single-topic dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON file. Defaults to <input stem>_expectedFailure.json.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(expectedFailure_MODELS),
        default="learningpotential",
        help="expectedFailure model to replay.",
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--initial-mastery", type=float, default=0.4)
    parser.add_argument("--ncc-window", type=int, default=2)
    parser.add_argument(
        "--recalculate-initial-step",
        action="store_true",
        help="Also update cumulative expectedFailure from step 0 instead of preserving the old frozen-initial-step behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.input.with_name(f"{args.input.stem}_expectedFailure.json")

    if output.resolve() == args.input.resolve():
        raise SystemExit(
            f"Refusing to overwrite {args.input}: recalculated trajectories only keep "
            "expectedFailure fields and drop mastery/action data. Pass a different --output path."
        )

    recalculate_file(
        input_path=args.input,
        output_path=output,
        dataset_path=args.dataset_file,
        topic=args.topic,
        model_name=args.model,
        lam=args.lam,
        beta=args.beta,
        initial_mastery=args.initial_mastery,
        ncc_window=args.ncc_window,
        keep_initial_step_frozen=not args.recalculate_initial_step,
    )
    print(f"Saved recalculated expectedFailure to {output}")


if __name__ == "__main__":
    main()
