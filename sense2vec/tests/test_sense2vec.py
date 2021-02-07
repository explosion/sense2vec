import pytest
import numpy
from sense2vec import Sense2Vec, registry


def test_sense2vec_object():
    s2v = Sense2Vec(shape=(10, 4))
    assert s2v.vectors.shape == (10, 4)
    assert len(s2v) == 10
    test_vector = numpy.asarray([4, 2, 2, 2], dtype=numpy.float32)
    s2v.add("test", test_vector)
    assert "test" in s2v
    assert isinstance(s2v.strings["test"], int)
    assert s2v.strings["test"] in s2v
    assert "foo" not in s2v
    assert numpy.array_equal(s2v["test"], test_vector)
    assert numpy.array_equal(s2v[s2v.strings["test"]], test_vector)
    assert list(s2v.keys()) == ["test"]
    s2v.add("test2", test_vector)
    assert "test2" in s2v
    assert sorted(list(s2v.keys())) == ["test", "test2"]
    with pytest.raises(ValueError):
        s2v["test3"] = test_vector
    s2v["test2"] = test_vector


def test_sense2vec_freqs():
    s2v = Sense2Vec(shape=(10, 4))
    vector = numpy.asarray([4, 2, 2, 2], dtype=numpy.float32)
    s2v.add("test1", vector, 123)
    s2v.add("test2", vector, 456)
    assert len(s2v.freqs) == 2
    assert s2v.get_freq("test1") == 123
    assert s2v.get_freq("test2") == 456
    assert s2v.get_freq("test3") is None
    assert s2v.get_freq("test3", 100) == 100
    s2v.set_freq("test3", 200)
    assert s2v.get_freq("test3") == 200


def test_sense2vec_other_senses():
    s2v = Sense2Vec(shape=(6, 4))
    s2v.cfg["senses"] = ["A", "B", "C", "D"]
    for key in ["a|A", "a|B", "a|C", "b|A", "b|C", "c|A"]:
        s2v.add(key, numpy.asarray([4, 2, 2, 2], dtype=numpy.float32))
    others = s2v.get_other_senses("a|A")
    assert sorted(others) == ["a|B", "a|C"]
    others = s2v.get_other_senses("b|C")
    assert others == ["b|A"]
    others = s2v.get_other_senses("c|A")
    assert others == []


def test_sense2vec_best_sense():
    s2v = Sense2Vec(shape=(5, 4))
    s2v.cfg["senses"] = ["A", "B", "C"]
    for key, freq in [("a|A", 100), ("a|B", 50), ("a|C", 10), ("b|A", 1), ("B|C", 2)]:
        s2v.add(key, numpy.asarray([4, 2, 2, 2], dtype=numpy.float32), freq)
    assert s2v.get_best_sense("a") == "a|A"
    assert s2v.get_best_sense("b") == "B|C"
    assert s2v.get_best_sense("b", ignore_case=False) == "b|A"
    assert s2v.get_best_sense("c") is None
    s2v.cfg["senses"] = []
    assert s2v.get_best_sense("a") is None
    assert s2v.get_best_sense("b", ["A"]) == "b|A"
    assert s2v.get_best_sense("b", ["A", "C"]) == "B|C"


def test_sense2vec_similarity():
    s2v = Sense2Vec(shape=(5, 4))
    s2v.add("a", numpy.asarray([4, 4, 2, 2], dtype=numpy.float32))
    s2v.add("b", numpy.asarray([4, 4, 2, 2], dtype=numpy.float32))
    s2v.add("c", numpy.asarray([4, 4, 4, 2], dtype=numpy.float32))
    s2v.add("d", numpy.asarray([0.1, 0.2, 0.3, 0.4], dtype=numpy.float32))
    s2v.add("e", numpy.asarray([0, 0, 0, 0], dtype=numpy.float32))
    assert s2v.similarity("a", "b") == 1.0
    assert 1.0 > s2v.similarity("b", "c") > 0.9
    assert 1.0 > s2v.similarity(["a", "b"], "c") > 0.9
    assert s2v.similarity("b", "c") == s2v.similarity(["a", "b"], "c")
    assert s2v.similarity("a", "d") < 0.8
    assert s2v.similarity("a", "e") == 0.0


def test_sense2vec_most_similar():
    s2v = Sense2Vec(shape=(6, 4))
    s2v.add("a", numpy.asarray([4, 2, 2, 2], dtype=numpy.float32))
    s2v.add("b", numpy.asarray([4, 4, 2, 2], dtype=numpy.float32))
    s2v.add("c", numpy.asarray([4, 4, 4, 2], dtype=numpy.float32))
    s2v.add("d", numpy.asarray([4, 4, 4, 4], dtype=numpy.float32))
    s2v.add("x", numpy.asarray([4, 2, 2, 2], dtype=numpy.float32))
    s2v.add("y", numpy.asarray([0.1, 1, 1, 1], dtype=numpy.float32))
    result1 = s2v.most_similar(["x"], n=2)
    assert len(result1) == 2
    assert result1[0][0] == "a"
    assert result1[0][1] == 1.0
    assert result1[0][1] == pytest.approx(1.0)
    assert result1[1][0] == "b"
    result2 = s2v.most_similar(["a", "x"], n=2)
    assert len(result2) == 2
    assert sorted([key for key, _ in result2]) == ["b", "d"]
    result3 = s2v.most_similar(["a", "b"], n=3)
    assert len(result3) == 3
    assert "y" not in [key for key, _ in result3]
    assert len(s2v.most_similar(["a", "b"], n=10)) == 4
    with pytest.raises(ValueError):
        s2v.most_similar(["z"], n=1)  # key not in table


def test_sense2vec_to_from_bytes():
    s2v = Sense2Vec(shape=(2, 4))
    test_vector1 = numpy.asarray([1, 2, 3, 4], dtype=numpy.float32)
    test_vector2 = numpy.asarray([5, 6, 7, 8], dtype=numpy.float32)
    s2v.add("test1", test_vector1, 123)
    s2v.add("test2", test_vector2, 456)
    s2v_bytes = s2v.to_bytes()
    new_s2v = Sense2Vec().from_bytes(s2v_bytes)
    assert len(new_s2v) == 2
    assert new_s2v.vectors.shape == (2, 4)
    assert "test1" in new_s2v
    assert "test2" in new_s2v
    assert new_s2v.get_freq("test1") == 123
    assert new_s2v.get_freq("test2") == 456
    assert numpy.array_equal(new_s2v["test1"], test_vector1)
    assert numpy.array_equal(new_s2v["test2"], test_vector2)
    assert s2v_bytes == new_s2v.to_bytes()
    s2v_bytes2 = s2v.to_bytes(exclude=["strings"])
    new_s2v2 = Sense2Vec().from_bytes(s2v_bytes2)
    assert len(new_s2v2.strings) == 0
    assert "test1" in new_s2v2
    assert s2v.strings["test1"] in new_s2v2
    with pytest.raises(KeyError):  # can't resolve hash
        new_s2v2.strings[s2v.strings["test2"]]


def test_registry():
    """Test that custom functions are used internally if they're registered."""

    @registry.make_key.register("custom_make_key")
    def custom_make_key(word, sense):
        return f"{word}###{sense}"

    @registry.split_key.register("custom_split_key")
    def custom_split_key(key):
        return tuple(key.split("###"))

    overrides = {"make_key": "custom_make_key", "split_key": "custom_split_key"}
    test_vector = numpy.asarray([1, 2, 3, 4], dtype=numpy.float32)
    data = [("clear", "NOUN", 100), ("clear", "VERB", 200), ("clear", "ADJ", 300)]
    s2v = Sense2Vec(shape=(len(data), 4), overrides=overrides)
    for word, sense, freq in data:
        s2v.add(custom_make_key(word, sense), test_vector, freq)
        s2v.cfg["senses"].append(sense)
    assert "clear###NOUN" in s2v
    other_senses = s2v.get_other_senses("clear###NOUN")
    assert len(other_senses) == 2
    assert "clear###VERB" in other_senses
    assert "clear###ADJ" in other_senses
    assert s2v.get_best_sense("clear") == "clear###ADJ"
