from typing import Union, Callable, List, Tuple
import re
from spacy.tokens import Doc, Token, Span
from spacy.util import filter_spans


DEFAULT_SENSE = "?"


def make_key(word: str, sense: str) -> str:
    """Create a key from a word and sense, e.g. "usage_example|NOUN".

    word (unicode): The word.
    sense (unicode): The sense.
    RETURNS (unicode): The key.
    """
    text = re.sub(r"\s", "_", word)
    return text + "|" + sense


def split_key(key: str) -> Tuple[str, str]:
    """Split a key into word and sense, e.g. ("usage example", "NOUN").

    key (unicode): The key to split.
    RETURNS (tuple): The split (word, sense) tuple.
    """
    word, sense = key.replace("_", " ").rsplit("|", 1)
    return word, sense


def make_spacy_key(
    obj: Union[Token, Span],
    make_key: Callable[[str, str], str] = make_key,
    prefer_ents: bool = False,
) -> str:
    """Create a key from a spaCy object, i.e. a Token or Span. If the object
    is a token, the part-of-speech tag (Token.pos_) is used for the sense
    and a special string is created for URLs. If the object is a Span and
    has a label (i.e. is an entity span), the label is used. Otherwise, the
    span's root part-of-speech tag becomes the sense.

    obj (Token / Span): The spaCy object to create the key for.
    make_key (callable): function that takes a word and sense string and
        creates the key (e.g. "word|sense").
    prefer_ents (bool): Prefer entity types for single tokens (i.e.
        token.ent_type instead of tokens.pos_). Should be enabled if phrases
        are merged into single tokens, because otherwise the entity sense would
        never be used.
    RETURNS (unicode): The key.
    """
    text = obj.text
    if isinstance(obj, Token):
        if obj.like_url:
            text = "%%URL"
            sense = "X"
        elif obj.ent_type_ and prefer_ents:
            sense = obj.ent_type_
        else:
            sense = obj.pos_
    elif isinstance(obj, Span):
        sense = obj.label_ or obj.root.pos_
    return make_key(text, sense or DEFAULT_SENSE)


def get_phrases(doc: Doc) -> List[Span]:
    """Compile a list of sense2vec phrases based on a processed Doc: named
    entities and noun chunks without determiners.

    doc (Doc): The Doc to get phrases from.
    RETURNS (list): The phrases as a list of Span objects.
    """
    spans = list(doc.ents)
    ent_words = set()
    for span in spans:
        ent_words.update(token.i for token in span)
    if doc.is_parsed:
        for np in doc.noun_chunks:
            # Prefer entities over noun chunks if there's overlap.
            if not any(w.i in ent_words for w in np):
                while len(np) > 1 and np[0].dep_ not in ("advmod", "amod", "compound"):
                    np = np[1:]
                spans.append(np)
    return spans


def merge_phrases(doc: Doc) -> Doc:
    """Transform a spaCy Doc to match the sense2vec format: merge entities
    into one token and merge noun chunks without determiners.

    doc (Doc): The document to merge phrases in.
    RETURNS (Doc): The Doc with merged tokens.
    """
    spans = get_phrases(doc)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    return doc
