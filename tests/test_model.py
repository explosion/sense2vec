import pytest
from pathlib import Path
from sense2vec import Sense2Vec
import numpy


@pytest.fixture
def s2v():
    data_path = Path(__file__).parent / "data"
    return Sense2Vec().from_disk(data_path)


def test_model_most_similar(s2v):
    s2v.cache = None
    assert "beekeepers|NOUN" in s2v
    ((key1, _), (key2, _)) = s2v.most_similar(["beekeepers|NOUN"], n=2)
    assert key1 == "honey_bees|NOUN"
    assert key2 == "Beekeepers|NOUN"


def test_model_most_similar_cache(s2v):
    query = "beekeepers|NOUN"
    assert s2v.cache
    assert query in s2v
    # Modify cache to test that the cache is used and values aren't computed
    query_row = s2v.vectors.find(key=s2v.ensure_int_key(query))
    scores = numpy.array(s2v.cache["scores"], copy=True)  # otherwise not writable
    scores[query_row, 1] = 2.0
    scores[query_row, 2] = 3.0
    s2v.cache["scores"] = scores
    ((key1, score1), (key2, score2)) = s2v.most_similar([query], n=2)
    assert key1 == "honey_bees|NOUN"
    assert score1 == 2.0
    assert key2 == "Beekeepers|NOUN"
    assert score2 == 3.0


def test_model_other_senses(s2v):
    others = s2v.get_other_senses("duck|NOUN")
    assert len(others) == 1
    assert others[0] == "duck|VERB"


def test_model_best_sense(s2v):
    assert s2v.get_best_sense("duck") == "duck|NOUN"
    assert s2v.get_best_sense("honey bees") == "honey_bees|NOUN"
