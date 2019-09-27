# coding: utf8
from __future__ import unicode_literals

import pytest
import numpy
from sense2vec import Sense2Vec


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


def test_sense2vec_most_similar():
    s2v = Sense2Vec(shape=(6, 4))
    s2v.add("a", numpy.asarray([4, 2, 2, 2], dtype=numpy.float32))
    s2v.add("b", numpy.asarray([4, 4, 2, 2], dtype=numpy.float32))
    s2v.add("c", numpy.asarray([4, 4, 4, 2], dtype=numpy.float32))
    s2v.add("d", numpy.asarray([4, 4, 4, 4], dtype=numpy.float32))
    s2v.add("x", numpy.asarray([4, 2, 2, 2], dtype=numpy.float32))
    s2v.add("y", numpy.asarray([0.1, 1, 1, 1], dtype=numpy.float32))
    result1 = s2v.most_similar(["x"])
    assert len(result1)
    assert result1[0][0] == "a"
    # assert result1[0][1] == 1.0
    result2 = s2v.most_similar(["y"])
    assert len(result2) == 0


def test_sense2vec_to_from_bytes():
    s2v = Sense2Vec(shape=(2, 4))
    test_vector1 = numpy.asarray([1, 2, 3, 4], dtype=numpy.float32)
    test_vector2 = numpy.asarray([5, 6, 7, 8], dtype=numpy.float32)
    s2v.add("test1", test_vector1)
    s2v.add("test2", test_vector2)
    s2v_bytes = s2v.to_bytes()
    new_s2v = Sense2Vec().from_bytes(s2v_bytes)
    assert len(new_s2v) == 2
    assert new_s2v.vectors.shape == (2, 4)
    assert "test1" in new_s2v
    assert "test2" in new_s2v
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