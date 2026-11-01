from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluate_retrieval import load_fixture


def test_evaluation_fixture_has_labeled_queries():
    fixture = load_fixture()

    assert len(fixture) >= 4
    assert all(item["query"] for item in fixture)
    assert all(item["source"] for item in fixture)
    assert len({item["source"] for item in fixture}) == len(fixture)
