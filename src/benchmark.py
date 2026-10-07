from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TypedDict
import warnings


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "MATHE"
    / "processed"
    / "Mathe_QA_preprocessed_full_one_topic.json"
)


def dataset_name(dataset_path: str | Path) -> str:
    """Return a filesystem-safe dataset name derived from its path."""
    path = Path(dataset_path)
    if path.parent.name.casefold() == "processed":
        name = path.parent.parent.name
    else:
        name = path.stem
        for suffix in ("_preprocessed", "-preprocessed"):
            if name.casefold().endswith(suffix):
                name = name[: -len(suffix)]
                break
    return _filename_slug(name)


def trajectory_filename(dataset_path: str | Path, topic: str) -> str:
    """Build ``<dataset>_<topic>_trajectories.json`` for a simulation run."""
    return f"{dataset_name(dataset_path)}_{_filename_slug(topic)}_trajectories.json"


def _filename_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not slug:
        raise ValueError(f"Cannot derive a filename component from {value!r}.")
    return slug


class RawQuestion(TypedDict):
    """Fields required for every question in a preprocessed dataset JSON."""

    topic: str
    difficulty: float


class Question(TypedDict):
    """Minimal question representation used by the recommendation algorithm."""

    question_index: int
    item_id: int
    topic: str
    difficulty: float


@dataclass(frozen=True)
class QuestionDataset:
    questions: list[Question]
    skills: list[str]
    question_skill_map: dict[int, dict[str, Any]]
    item_question_map: dict[int, Question]
    questions_by_difficulty: dict[
        str, dict[float, tuple[Question, ...]]
    ] = field(init=False, repr=False, compare=False)
    difficulty_levels: dict[str, tuple[float, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Index each topic once for fast adjacent-difficulty lookups."""
        mutable_groups: dict[str, dict[float, list[Question]]] = {}
        for question in self.questions:
            topic = question["topic"]
            difficulty = float(question["difficulty"])
            mutable_groups.setdefault(topic, {}).setdefault(
                difficulty, []
            ).append(question)

        groups = {
            topic: {
                difficulty: tuple(group)
                for difficulty, group in difficulty_groups.items()
            }
            for topic, difficulty_groups in mutable_groups.items()
        }
        levels = {
            topic: tuple(sorted(difficulty_groups))
            for topic, difficulty_groups in groups.items()
        }
        object.__setattr__(self, "questions_by_difficulty", groups)
        object.__setattr__(self, "difficulty_levels", levels)

    @property
    def irt_difficulties(self) -> dict[int, float]:
        return {
            int(question["question_index"]): float(question["difficulty"])
            for question in self.questions
        }

    def get_question(self, question_index: int) -> Question:
        return self.questions[int(question_index)]

    def get_question_by_item(self, item_id: int) -> Question:
        """Return the simulator question corresponding to an answer-log item."""
        try:
            return self.item_question_map[int(item_id)]
        except KeyError as error:
            raise ValueError(
                f"Answer item {item_id!r} is not present in the question dataset."
            ) from error

    def resolve_topic(self, requested_topic: str | None = None) -> str:
        """Resolve an explicit topic or infer it for a single-topic dataset."""
        if requested_topic is not None:
            if requested_topic not in self.skills:
                raise ValueError(
                    f"Unknown topic {requested_topic!r}. Available topics: {self.skills}"
                )
            return requested_topic
        if len(self.skills) != 1:
            raise ValueError(
                "The dataset contains multiple topics. Select one with --topic. "
                f"Available topics: {self.skills}"
            )
        return self.skills[0]

    def get_random_batch(
        self,
        size_k: int,
        topic: str,
        filter_question: Iterable[int] | None = None,
    ) -> list[Question]:
        """Return a random batch of questions for one skill."""
        filtered_indices = set(filter_question or [])
        if size_k <= 0:
            raise ValueError(f"Error -> size_k must be positive, got {size_k}")
        if topic not in self.skills:
            raise ValueError(f"Error -> {topic} not in skills list: {self.skills}")

        matching_questions = [
            question
            for question in self.questions
            if question["topic"] == topic
            and question["question_index"] not in filtered_indices
        ]

        if size_k > len(matching_questions):
            warnings.warn(f"Error -> requested {size_k} questions for {topic}, but only {len(matching_questions)} are available | Adding questions previously filtered")
            added_questions = [
                question
                for question in self.questions
                if question["topic"] == topic
            ]
            added_questions = random.sample(added_questions, k=size_k-len(matching_questions))
            matching_questions.extend(added_questions)


        if size_k > len(matching_questions):
            raise ValueError(
                f"Error -> requested {size_k} questions for {topic}, "
                f"but only {len(matching_questions)} are available"
            )

        return random.sample(matching_questions, k=size_k)
    
    def get_question_inferior_dif(
        self,
        question: Question,
        filter_question: Iterable[int] | None = None,
    ) -> Question:
        return self._get_question_at_adjacent_difficulty(
            question,
            filter_question=filter_question,
            direction=-1,
        )
    
    def get_question_superior_dif(
        self,
        question: Question,
        filter_question: Iterable[int] | None = None,
    ) -> Question:
        return self._get_question_at_adjacent_difficulty(
            question,
            filter_question=filter_question,
            direction=1,
        )

    def _get_question_at_adjacent_difficulty(
        self,
        question: Question,
        filter_question: Iterable[int] | None,
        direction: int,
    ) -> Question:
        """Choose randomly from the nearest available difficulty group."""
        difficulty = float(question["difficulty"])
        topic = question["topic"]
        filtered_indices = (
            filter_question if isinstance(filter_question, set) else set(filter_question or [])
        )
        levels = self.difficulty_levels.get(topic, ())
        groups = self.questions_by_difficulty.get(topic, {})

        if direction < 0:
            level_index = bisect_left(levels, difficulty) - 1
        else:
            level_index = bisect_right(levels, difficulty)

        while 0 <= level_index < len(levels):
            level = levels[level_index]
            available_questions = [
                candidate
                for candidate in groups[level]
                if candidate["question_index"] not in filtered_indices
            ]
            if available_questions:
                return random.choice(available_questions)
            level_index += direction

        relation = "below" if direction < 0 else "above"
        warnings.warn(
            f"Warning: No unfiltered question found {relation} difficulty "
            f"{difficulty} for skill {topic} | Returning initial question"
        )
        return question

    def get_random_batch_with_difficulty(
        self,
        size_k: int,
        topic: str,
        difficulty: float,
        filter_question: Iterable[int] | None = None,
        difficulty_window: float = 0.25,
    ) -> list[Question]:
        filtered_indices = set(filter_question or [])
        if size_k <= 0:
            raise ValueError(f"Error -> size_k must be positive, got {size_k}")
        if topic not in self.skills:
            raise ValueError(f"Error -> {topic} not in skills list: {self.skills}")

        matching_questions = [
            question
            for question in self.questions
            if question["topic"] == topic
            and difficulty - difficulty_window
            <= question["difficulty"]
            <= difficulty + difficulty_window
            and question["question_index"] not in filtered_indices
        ]
        if size_k > len(matching_questions):
            warnings.warn(f"Error -> requested {size_k} questions for {topic}, but only {len(matching_questions)} are available | Adding questions previously filtered")
            added_questions = [
                question
                for question in self.questions
                if question["topic"] == topic
                and difficulty - difficulty_window
                <= question["difficulty"]
                <= difficulty + difficulty_window
            ]
            added_questions = random.sample(added_questions, k=size_k-len(matching_questions))
            matching_questions.extend(added_questions)

        if size_k > len(matching_questions):
            raise ValueError(
                f"Error -> requested {size_k} questions for {topic}, "
                f"but only {len(matching_questions)} are available"
            )
        
        return random.sample(matching_questions, k=size_k)
        
    def get_initial_questions(
        self,
        size_k: int,
        topic: str,
        filter_question: Iterable[int] | None = None,
    ) -> list[Question]:
        """Select unique starting questions evenly across a topic's difficulty range."""
        filtered_indices = set(filter_question or [])
        if size_k <= 0:
            raise ValueError(f"Error -> size_k must be positive, got {size_k}")
        if topic not in self.skills:
            raise ValueError(f"Error -> {topic} not in skills list: {self.skills}")

        available_questions = [
            question
            for question in self.questions
            if question["topic"] == topic
            and question["question_index"] not in filtered_indices
        ]
        if size_k > len(available_questions):
            raise ValueError(
                f"Error -> requested {size_k} initial questions for {topic}, "
                f"but only {len(available_questions)} are available"
            )

        minimum_difficulty = min(
            question["difficulty"] for question in available_questions
        )
        maximum_difficulty = max(
            question["difficulty"] for question in available_questions
        )
        if size_k == 1:
            target_difficulties = [
                (minimum_difficulty + maximum_difficulty) / 2.0
            ]
        else:
            difficulty_step = (
                maximum_difficulty - minimum_difficulty
            ) / (size_k - 1)
            target_difficulties = [
                minimum_difficulty + index * difficulty_step
                for index in range(size_k)
            ]

        selected_questions: list[Question] = []
        remaining_questions = available_questions.copy()
        for target_difficulty in target_difficulties:
            nearest_distance = min(
                abs(question["difficulty"] - target_difficulty)
                for question in remaining_questions
            )
            nearest_questions = [
                question
                for question in remaining_questions
                if abs(question["difficulty"] - target_difficulty)
                == nearest_distance
            ]
            selected_question = random.choice(nearest_questions)
            selected_questions.append(selected_question)
            remaining_questions.remove(selected_question)

        return selected_questions


def load_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    target_topics: Iterable[str] | None = None,
    limit: int | None = None,
) -> QuestionDataset:
    """Load a preprocessed question dataset.

    Each JSON question must contain ``topic`` as a non-empty string and
    ``difficulty`` as a numeric value. Its source ``id`` is retained as
    ``item_id`` so historical answers can be resolved; when absent, the source
    row index is used. The JSON root may be a question list or an object with a
    ``questions`` list.
    """
    dataset_path = Path(dataset_path)

    raw_questions = _load_raw_questions(dataset_path)
    selected_topics = set(target_topics or [])

    filtered_raw_questions = []
    for raw_index, raw_question in enumerate(raw_questions):
        topic, difficulty = _required_question_fields(
            raw_question,
            question_index=raw_index,
            dataset_path=dataset_path,
        )
        if selected_topics and topic not in selected_topics:
            continue
        if limit is not None and len(filtered_raw_questions) >= limit:
            break
        item_id = _question_item_id(
            raw_question,
            question_index=raw_index,
            dataset_path=dataset_path,
        )
        filtered_raw_questions.append((item_id, topic, difficulty))
    if not filtered_raw_questions:
        target_detail = ", ".join(sorted(selected_topics)) or "all topics"
        raise ValueError(f"No questions found for {target_detail} in {dataset_path}.")

    questions: list[Question] = []
    item_question_map: dict[int, Question] = {}
    for question_index, (item_id, topic, difficulty) in enumerate(
        filtered_raw_questions
    ):
        if item_id in item_question_map:
            raise ValueError(
                f"Duplicate question item ID {item_id} in {dataset_path}."
            )
        question: Question = {
            "question_index": question_index,
            "item_id": item_id,
            "topic": topic,
            "difficulty": difficulty,
        }
        questions.append(question)
        item_question_map[item_id] = question

    skills, question_skill_map = create_question_skill_map(questions)
    return QuestionDataset(
        questions=questions,
        skills=skills,
        question_skill_map=question_skill_map,
        item_question_map=item_question_map,
    )


def create_question_skill_map(
    questions: list[Question],
) -> tuple[list[str], dict[int, dict[str, Any]]]:
    skill_set: set[str] = set()
    question_skill_map: dict[int, dict[str, Any]] = {}

    for question in questions:
        question_index = int(question["question_index"])
        skills = []
        topic = question["topic"]
        if topic:
            skills.append(str(topic))
            skill_set.add(str(topic))

        question_skill_map[question_index] = {
            "skills": skills,
            "difficulty": float(question["difficulty"]),
        }

    return sorted(skill_set), question_skill_map


def _load_raw_questions(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return list(payload["questions"])
    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unsupported dataset JSON format: {dataset_path}")


def _required_question_fields(
    raw_question: dict[str, Any],
    question_index: int,
    dataset_path: Path,
) -> tuple[str, float]:
    """Validate and return the required ``topic`` and ``difficulty`` fields."""
    if not isinstance(raw_question, dict):
        raise ValueError(
            f"Question {question_index} in {dataset_path} must be a JSON object."
        )

    missing_fields = [
        field_name
        for field_name in RawQuestion.__required_keys__
        if field_name not in raw_question
    ]
    if missing_fields:
        raise ValueError(
            f"Question {question_index} in {dataset_path} is missing required fields: "
            f"{', '.join(sorted(missing_fields))}."
        )

    topic = raw_question["topic"]
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(
            f"Question {question_index} in {dataset_path} has invalid topic: {topic!r}."
        )

    difficulty_value = raw_question["difficulty"]
    if isinstance(difficulty_value, bool):
        raise ValueError(
            f"Question {question_index} in {dataset_path} has invalid difficulty: "
            f"{difficulty_value!r}."
        )
    try:
        difficulty = float(difficulty_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Question {question_index} in {dataset_path} has invalid difficulty: "
            f"{difficulty_value!r}."
        ) from error

    return topic.strip(), difficulty


def _question_item_id(
    raw_question: dict[str, Any],
    question_index: int,
    dataset_path: Path,
) -> int:
    """Return the source item ID used by the historical answer file."""
    item_id = raw_question.get("id", question_index)
    if isinstance(item_id, bool):
        raise ValueError(
            f"Question {question_index} in {dataset_path} has invalid id: "
            f"{item_id!r}."
        )
    try:
        return int(item_id)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Question {question_index} in {dataset_path} has invalid id: "
            f"{item_id!r}."
        ) from error
