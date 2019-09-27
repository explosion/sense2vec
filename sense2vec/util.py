from typing import Union, Callable, List, Tuple
import re
from spacy.tokens import Doc, Token, Span
from spacy.util import filter_spans


DEFAULT_SENSE = "?"


def merge_phrases(doc: Doc) -> Doc:
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


def make_key(word: str, sense: str) -> str:
    text = re.sub(r"\s", "_", word)
    return text + "|" + sense


def split_key(key: str) -> Tuple[str, str]:
    word, sense = key.replace("_", " ").rsplit("|", 1)
    return word, sense


def make_spacy_key(
    obj: Union[Token, Span], make_key: Callable[[str, str], str] = make_key
) -> str:
    text = obj.text
    if isinstance(obj, Token):
        if obj.like_url:
            text = "%%URL"
            sense = "X"
        else:
            sense = obj.pos_
    elif isinstance(obj, Span):
        sense = obj.label_ or obj.root.pos_
    return make_key(text, sense or DEFAULT_SENSE)


def get_phrases(doc: Doc) -> List[Span]:
    spans = list(doc.ents)
    if doc.is_parsed:
        for np in doc.noun_chunks:
            while len(np) > 1 and np[0].dep_ not in ("advmod", "amod", "compound"):
                np = np[1:]
            spans.append(np)
    return spans
