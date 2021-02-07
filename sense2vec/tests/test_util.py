import pytest
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from sense2vec.util import get_true_cased_text, make_key, split_key


def get_doc(vocab, words, spaces, pos):
    doc = Doc(vocab, words=words, spaces=spaces)
    for i, pos_tag in enumerate(pos):
        doc[i].pos_ = pos_tag
    return doc


def test_get_true_cased_text():
    vocab = Vocab()
    words1 = ["Cool", ",", "thanks", "!"]
    spaces1 = [False, True, False, False]
    pos1 = ["ADJ", "PUNCT", "NOUN", "PUNCT"]
    doc1 = get_doc(vocab, words1, spaces1, pos1)
    assert get_true_cased_text(doc1[0:4]) == "cool, thanks!"
    assert get_true_cased_text(doc1[0]) == "cool"
    assert get_true_cased_text(doc1[2:4]) == "thanks!"
    words2 = ["I", "can", "understand", "."]
    spaces2 = [True, True, False, False]
    pos2 = ["PRON", "VERB", "VERB", "PUNCT"]
    doc2 = get_doc(vocab, words2, spaces2, pos2)
    assert get_true_cased_text(doc2[0:4]) == "I can understand."
    assert get_true_cased_text(doc2[0]) == "I"
    assert get_true_cased_text(doc2[2:4]) == "understand."
    words3 = ["You", "think", "Obama", "was", "pretty", "good", "..."]
    spaces3 = [True, True, True, True, True, False, False]
    pos3 = ["PRON", "VERB", "PROPN", "VERB", "ADV", "ADJ", "PUNCT"]
    doc3 = get_doc(vocab, words3, spaces3, pos3)
    doc3.ents = [Span(doc3, 2, 3, label="PERSON")]
    assert get_true_cased_text(doc3[0:7]) == "You think Obama was pretty good..."
    assert get_true_cased_text(doc3[0]) == "you"
    assert get_true_cased_text(doc3[2]) == "Obama"
    assert get_true_cased_text(doc3[4:6]) == "pretty good"
    words4 = ["Ok", ",", "Barack", "Obama", "was", "pretty", "good", "..."]
    spaces4 = [False, True, True, True, True, True, False, False]
    pos4 = ["INTJ", "PUNCT", "PROPN", "PROPN", "VERB", "ADV", "ADJ", "PUNCT"]
    doc4 = get_doc(vocab, words4, spaces4, pos4)
    doc4.ents = [Span(doc4, 2, 4, label="PERSON")]
    assert get_true_cased_text(doc4[0:8]) == "Ok, Barack Obama was pretty good..."
    assert get_true_cased_text(doc4[2:4]) == "Barack Obama"
    assert get_true_cased_text(doc4[3]) == "Obama"


@pytest.mark.parametrize(
    "word,sense,expected",
    [
        ("foo", "bar", "foo|bar"),
        ("hello world", "TEST", "hello_world|TEST"),
        ("hello world |test!", "TEST", "hello_world_|test!|TEST"),
    ],
)
def test_make_split_key(word, sense, expected):
    assert make_key(word, sense) == expected
    assert split_key(expected) == (word, sense)
