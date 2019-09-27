# coding: utf8
from __future__ import unicode_literals

from spacy.tokens import Token, Span
from spacy.util import filter_spans


def transform_doc(doc):
    """
    Transform a spaCy Doc to match the sense2vec format: merge entities
    into one token and merge noun chunks without determiners.
    """
    spans = get_phrases(doc)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            root = span.root
            attrs = {"tag": root.tag_, "lemma": root.lemma_, "ent_type": root.ent_type_}
            retokenizer.merge(span, attrs=attrs)
    return doc


def make_key(obj):
    text = obj.text.replace(" ", "_")
    if isinstance(obj, Token):
        return text + "|" + obj.pos_
    elif isinstance(obj, Span):
        if obj.label_:
            return text + "|" + obj.label_
        return text + "|" + obj.root.pos_
    return text


def split_key(key):
    return tuple(key.replace("_", " ").rsplit("|", 1))


def make_token_key(token):
    return token.text.replace(" ", "_") + "|" + token.pos_


def make_span_key(span):
    text = span.text.replace(" ", "_")
    if span.label_:
        return text + "|" + span.label_
    return text + "|" + span.root.pos_


def get_phrases(doc):
    spans = list(doc.ents)
    if doc.is_parsed:
        for np in doc.noun_chunks:
            while len(np) > 1 and np[0].dep_ not in ("advmod", "amod", "compound"):
                np = np[1:]
            spans.append(np)
    return spans
