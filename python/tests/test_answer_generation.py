import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from answer_generation import MockAnswerGenerator, build_grounded_prompt, generate_grounded_answer
from context_builder import ContextPackage


def sample_context() -> ContextPackage:
    return ContextPackage(
        query="What does the engine store?",
        retrieval={"selected_count": 1},
        evidence=[
            {
                "citation": "[1] docs/storage.md · memory:storage-1",
                "text": "The engine stores memory content and embeddings.",
            }
        ],
        context_text="[1] docs/storage.md · memory:storage-1\nThe engine stores memory content and embeddings.",
    )


def test_mock_generator_returns_grounded_demo_answer():
    context = sample_context()

    answer = generate_grounded_answer(context, MockAnswerGenerator())

    assert "memory content and embeddings" in answer
    assert "[1]" in answer
    assert "only the evidence" in build_grounded_prompt(context)

