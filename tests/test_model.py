import pytest
from pathlib import Path
from sense2vec import Sense2Vec


@pytest.fixture
def s2v():
    data_path = Path(__file__).parent / "data"
    return Sense2Vec().from_disk(data_path)


def test_model_most_similar(s2v):
    assert "beekeepers|NOUN" in s2v
    result = s2v.most_similar(["beekeepers|NOUN"], n=2)
    assert result[0][0] == "honey_bees|NOUN"
    assert result[1][0] == "Beekeepers|NOUN"


def test_model_other_senses(s2v):
    others = s2v.get_other_senses("duck|NOUN")
    assert len(others) == 1
    assert others[0] == "duck|VERB"


def test_model_best_sense(s2v):
    assert s2v.get_best_sense("duck") == "duck|NOUN"
    assert s2v.get_best_sense("honey bees") == "honey_bees|NOUN"
