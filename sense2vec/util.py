from typing import Union, List, Tuple, Set
import re
from spacy.tokens import Doc, Token, Span
from spacy.util import filter_spans
import catalogue

try:
    import importlib.metadata as importlib_metadata  # Python 3.8
except ImportError:
    import importlib_metadata  # noqa: F401


class registry(object):
    make_key = catalogue.create("sense2vec", "make_key")
    split_key = catalogue.create("sense2vec", "split_key")
    make_spacy_key = catalogue.create("sense2vec", "make_spacy_key")
    get_phrases = catalogue.create("sense2vec", "get_phrases")
    merge_phrases = catalogue.create("sense2vec", "merge_phrases")


@registry.make_key.register("default")
def make_key(word: str, sense: str) -> str:
    """Create a key from a word and sense, e.g. "usage_example|NOUN".

    word (unicode): The word.
    sense (unicode): The sense.
    RETURNS (unicode): The key.
    """
    text = re.sub(r"\s", "_", word)
    return text + "|" + sense


@registry.split_key.register("default")
def split_key(key: str) -> Tuple[str, str]:
    """Split a key into word and sense, e.g. ("usage example", "NOUN").

    key (unicode): The key to split.
    RETURNS (tuple): The split (word, sense) tuple.
    """
    if not isinstance(key, str) or "|" not in key:
        raise ValueError(f"Invalid key: {key}")
    word, sense = key.replace("_", " ").rsplit("|", 1)
    return word, sense


@registry.make_spacy_key.register("default")
def make_spacy_key(
    obj: Union[Token, Span], prefer_ents: bool = False, lemmatize: bool = False
) -> Tuple[str, str]:
    """Create a key from a spaCy object, i.e. a Token or Span. If the object
    is a token, the part-of-speech tag (Token.pos_) is used for the sense
    and a special string is created for URLs. If the object is a Span and
    has a label (i.e. is an entity span), the label is used. Otherwise, the
    span's root part-of-speech tag becomes the sense.

    obj (Token / Span): The spaCy object to create the key for.
    prefer_ents (bool): Prefer entity types for single tokens (i.e.
        token.ent_type instead of tokens.pos_). Should be enabled if phrases
        are merged into single tokens, because otherwise the entity sense would
        never be used.
    lemmatize (bool): Use the object's lemma instead of its text.
    RETURNS (unicode): The key.
    """
    default_sense = "?"
    text = get_true_cased_text(obj, lemmatize=lemmatize)
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
    return (text, sense or default_sense)


def get_true_cased_text(obj: Union[Token, Span], lemmatize: bool = False):
    """Correct casing so that sentence-initial words are not title-cased. Named
    entities and other special cases (such as the word "I") should still be
    title-cased.

    obj (Token / Span): The spaCy object to conver to text.
    lemmatize (bool): Use the object's lemma instead of its text.
    RETURNS (unicode): The converted text.
    """
    if lemmatize:
        return obj.lemma_
    if isinstance(obj, Token) and (not obj.is_sent_start or obj.ent_type_):
        return obj.text
    elif isinstance(obj, Span) and (not obj[0].is_sent_start or obj[0].ent_type):
        return obj.text
    elif (  # Okay, we have a non-entity, starting a sentence
        not obj.text[0].isupper()  # Is its first letter upper-case?
        or any(c.isupper() for c in obj.text[1:])  # ..Only its first letter?
        or obj.text[0] == "I"  # Is it "I"?
    ):
        return obj.text
    else:  # Fix the casing
        return obj.text.lower()


def get_noun_phrases(doc: Doc) -> List[Span]:
    """Compile a list of noun phrases in sense2vec's format (without
    determiners). Separated out to make it easier to customize, e.g. for
    languages that don't implement a noun_chunks iterator out-of-the-box, or
    use different label schemes.

    doc (Doc): The Doc to get noun phrases from.
    RETURNS (list): The noun phrases as a list of Span objects.
    """
    trim_labels = ("advmod", "amod", "compound")
    spans = []
    if doc.is_parsed:
        for np in doc.noun_chunks:
            while len(np) > 1 and np[0].dep_ not in trim_labels:
                np = np[1:]
            spans.append(np)
    return spans


@registry.get_phrases.register("default")
def get_phrases(doc: Doc) -> List[Span]:
    """Compile a list of sense2vec phrases based on a processed Doc: named
    entities and noun chunks without determiners.

    doc (Doc): The Doc to get phrases from.
    RETURNS (list): The phrases as a list of Span objects.
    """
    spans = list(doc.ents)
    ent_words: Set[str] = set()
    for span in spans:
        ent_words.update(token.i for token in span)
    for np in get_noun_phrases(doc):
        # Prefer entities over noun chunks if there's overlap
        if not any(w.i in ent_words for w in np):
            spans.append(np)
    return spans


def is_particle(
    token: Token, pos: Tuple[str] = ("PART",), deps: Tuple[str] = ("prt",)
) -> bool:
    """Determine whether a word is a particle, for phrasal verb detection.

    token (Token): The token to check.
    pos (tuple): The universal POS tags to check (Token.pos_).
    deps (tuple): The dependency labels to check (Token.dep_).
    """
    return token.pos_ in pos or token.dep_ in deps


@registry.merge_phrases.register("default")
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


class SimpleFrozenDict(dict):
    """Simplified implementation of a frozen dict, mainly used as default
    function or method argument (for arguments that should default to empty
    dictionary). Will raise an error if user or spaCy attempts to add to dict.
    """

    err = (
        "Can't write to frozen dictionary. This is likely an internal error. "
        "Are you writing to a default function argument?"
    )

    def __setitem__(self, key, value):
        raise NotImplementedError(self.err)

    def pop(self, key, default=None):
        raise NotImplementedError(self.err)

    def update(self, other):
        raise NotImplementedError(self.err)
