import pytest
import numpy

from fancyvec.vectors import Vectors


def test_init():
    vec = Vectors(128)
    assert vec.mem is not None
    with pytest.raises(AttributeError) as excinfo:
        vec.mem = None


def test_add():
    vecs = Vectors(128)
    good = numpy.ndarray(shape=(vecs.nr_dim,), dtype='float32')
    vecs.add(good)
    bad = numpy.ndarray(shape=(vecs.nr_dim+1,), dtype='float32')
    with pytest.raises(AssertionError) as excinfo:
        vecs.add(bad)


def test_borrow():
    vecs = Vectors(128)
    good = numpy.ndarray(shape=(vecs.nr_dim,), dtype='float32')
    vecs.borrow(good)
    bad = numpy.ndarray(shape=(vecs.nr_dim+1,), dtype='float32')
    with pytest.raises(AssertionError) as excinfo:
        vecs.borrow(bad)


def test_most_similar():
    vecs = Vectors(4)
    vecs.add(numpy.asarray([4,2,2,2], dtype='float32'))
    vecs.add(numpy.asarray([4,4,2,2], dtype='float32'))
    vecs.add(numpy.asarray([4,4,4,2], dtype='float32'))
    vecs.add(numpy.asarray([4,4,4,4], dtype='float32'))

    indices, scores = vecs.most_similar(
        numpy.asarray([4,2,2,2], dtype='float32'), 4)
    print(list(scores))
    assert list(indices) == [0,1]
    indices, scores = vecs.most_similar(
        numpy.asarray([0.1,1,1,1], dtype='float32'), 4)
    assert list(indices) == [4,3]
