import pytest
from pathlib import Path
from sense2vec import Sense2Vec


@pytest.fixture
def s2v():
    data_path = Path(__file__).parent / "data"
    return Sense2Vec().from_disk(data_path)


def test_most_similar(s2v):
    assert "beekeepers|NOUN" in s2v
    result = s2v.most_similar(["beekeepers|NOUN"])
