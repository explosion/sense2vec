import pytest
import numpy
from spacy.vocab import Vocab
from spacy.tokens import Doc, Span
from sense2vec import Sense2VecComponent


@pytest.fixture
def doc():
    vocab = Vocab()
    doc = Doc(vocab, words=["hello", "world"])
    doc[0].pos_ = "INTJ"
    doc[1].pos_ = "NOUN"
    return doc


def test_component_attributes(doc):
    s2v = Sense2VecComponent(doc.vocab, shape=(10, 4))
    vector = numpy.asarray([4, 2, 2, 2], dtype=numpy.float32)
    s2v.s2v.add("world|NOUN", vector, 123)
    doc = s2v(doc)
    assert doc[0]._.s2v_key == "hello|INTJ"
    assert doc[1]._.s2v_key == "world|NOUN"
    assert doc[0]._.in_s2v is False
    assert doc[1]._.in_s2v is True
    assert doc[0]._.s2v_freq is None
    assert doc[1]._.s2v_freq == 123
    assert numpy.array_equal(doc[1]._.s2v_vec, vector)


def test_component_attributes_ents(doc):
    s2v = Sense2VecComponent(doc.vocab, shape=(10, 4))
    s2v.first_run = False
    vector = numpy.asarray([4, 2, 2, 2], dtype=numpy.float32)
    s2v.s2v.add("world|NOUN", vector)
    s2v.s2v.add("world|GPE", vector)
    doc = s2v(doc)
    assert len(doc._.s2v_phrases) == 0
    doc.ents = [Span(doc, 1, 2, label="GPE")]
    assert len(doc._.s2v_phrases) == 1
    phrase = doc._.s2v_phrases[0]
    assert phrase._.s2v_key == "world|GPE"
    assert phrase[0]._.s2v_key == "world|NOUN"
    assert phrase._.in_s2v is True
    assert phrase[0]._.in_s2v is True


def test_component_similarity(doc):
    s2v = Sense2VecComponent(doc.vocab, shape=(4, 4))
    s2v.first_run = False
    vector = numpy.asarray([4, 2, 2, 2], dtype=numpy.float32)
    s2v.s2v.add("hello|INTJ", vector)
    s2v.s2v.add("world|NOUN", vector)
    doc = s2v(doc)
    assert doc[0]._.s2v_similarity(doc[1]) == 1.0
    assert doc[1:3]._.s2v_similarity(doc[1:3]) == 1.0


def test_component_lemmatize(doc):
    lookups = doc.vocab.lookups.add_table("lemma_lookup")
    lookups["world"] = "wrld"
    s2v = Sense2VecComponent(doc.vocab, shape=(4, 4), lemmatize=True)
    s2v.first_run = False
    vector = numpy.asarray([4, 2, 2, 2], dtype=numpy.float32)
    s2v.s2v.add("hello|INTJ", vector)
    s2v.s2v.add("world|NOUN", vector)
    s2v.s2v.add("wrld|NOUN", vector)
    doc = s2v(doc)
    assert doc[0]._.s2v_key == "hello|INTJ"
    assert doc[1].lemma_ == "wrld"
    assert doc[1]._.s2v_key == "wrld|NOUN"
    lookups["hello"] = "hll"
    assert doc[0].lemma_ == "hll"
    assert doc[0]._.s2v_key == "hello|INTJ"
    s2v.s2v.add("hll|INTJ", vector)
    assert doc[0]._.s2v_key == "hll|INTJ"
    new_s2v = Sense2VecComponent().from_bytes(s2v.to_bytes())
    assert new_s2v.s2v.cfg["lemmatize"] is True
    doc.vocab.lookups.remove_table("lemma_lookup")


def test_component_to_from_bytes(doc):
    s2v = Sense2VecComponent(doc.vocab, shape=(1, 4))
    s2v.first_run = False
    vector = numpy.asarray([4, 2, 2, 2], dtype=numpy.float32)
    s2v.s2v.add("world|NOUN", vector)
    assert "world|NOUN" in s2v.s2v
    assert "world|GPE" not in s2v.s2v
    doc = s2v(doc)
    assert doc[0]._.in_s2v is False
    assert doc[1]._.in_s2v is True
    s2v_bytes = s2v.to_bytes()
    new_s2v = Sense2VecComponent(doc.vocab).from_bytes(s2v_bytes)
    new_s2v.first_run = False
    assert "world|NOUN" in new_s2v.s2v
    assert numpy.array_equal(new_s2v.s2v["world|NOUN"], vector)
    assert "world|GPE" not in new_s2v.s2v
    new_s2v.s2v.vectors.resize((2, 4))
    new_s2v.s2v.add("hello|INTJ", vector)
    assert doc[0]._.in_s2v is False
    new_doc = new_s2v(doc)
    assert new_doc[0]._.in_s2v is True
