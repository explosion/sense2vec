from typing import Tuple, Union, List, Dict
from spacy import component
from spacy.tokens import Doc, Token, Span
from spacy.vocab import Vocab
from spacy.language import Language
from pathlib import Path
import numpy

from .sense2vec import Sense2Vec
from .util import registry, SimpleFrozenDict


@component(
    "sense2vec",
    requires=["token.pos", "token.dep", "token.ent_type", "token.ent_iob", "doc.ents"],
    assigns=[
        "doc._._s2v",
        "doc._.s2v_phrases",
        "token._.in_s2v",
        "token._.s2v_key",
        "token._.s2v_vec",
        "token._.s2v_freq",
        "token._.s2v_other_senses",
        "token._.s2v_most_similar",
        "token._.s2v_similarity",
        "span._.in_s2v",
        "span._.s2v_key",
        "span._.s2v_vec",
        "span._.s2v_freq",
        "span._.s2v_other_senses",
        "span._.s2v_most_similar",
        "span._.s2v_similarity",
    ],
)
class Sense2VecComponent(object):
    def __init__(
        self,
        vocab: Vocab = None,
        shape: Tuple[int, int] = (1000, 128),
        merge_phrases: bool = False,
        lemmatize: bool = False,
        overrides: Dict[str, str] = SimpleFrozenDict(),
        **kwargs,
    ):
        """Initialize the pipeline component.

        vocab (Vocab): The shared vocab. Mostly used for the shared StringStore.
        shape (tuple): The vector shape.
        merge_phrases (bool): Merge sense2vec phrases into one token.
        lemmatize (bool): Always look up lemmas if available in the vectors,
            otherwise default to original word.
        overrides (dict): Optional custom functions to use, mapped to names
            registered via the registry, e.g. {"make_key": "custom_make_key"}.
        RETURNS (Sense2VecComponent): The newly constructed object.
        """
        self.first_run = True
        self.merge_phrases = merge_phrases
        strings = vocab.strings if vocab is not None else None
        self.s2v = Sense2Vec(shape=shape, strings=strings)
        cfg = {
            "make_spacy_key": "default",
            "get_phrases": "default",
            "merge_phrases": "default",
            "lemmatize": lemmatize,
        }
        self.s2v.cfg.update(cfg)
        self.s2v.cfg.update(overrides)

    @classmethod
    def from_nlp(cls, nlp: Language, **cfg):
        """Initialize the component from an nlp object. Mostly used as the
        component factory for the entry point (see setup.cfg).

        nlp (Language): The nlp object.
        **cfg: Optional config parameters.
        RETURNS (Sense2VecComponent): The newly constructed object.
        """
        return cls(vocab=nlp.vocab, **cfg)

    def __call__(self, doc: Doc) -> Doc:
        """Process a Doc object with the component.

        doc (Doc): The document to process.
        RETURNS (Doc): The processed document.
        """
        if self.first_run:
            self.init_component()
            self.first_run = False
        # Store reference to s2v object on Doc to make sure it's right
        doc._._s2v = self.s2v
        if self.merge_phrases:
            merge_phrases = registry.merge_phrases.get(doc._._s2v.cfg["merge_phrases"])
            doc = merge_phrases(doc)
        return doc

    def init_component(self):
        """Register the component-specific extension attributes here and only
        if the component is added to the pipeline and used – otherwise, tokens
        will still get the attributes even if the component is only created and
        not added.
        """
        Doc.set_extension("_s2v", default=None)
        Doc.set_extension("s2v_phrases", getter=self.get_phrases)
        for obj in [Token, Span]:
            obj.set_extension("s2v_key", getter=self.s2v_key)
            obj.set_extension("in_s2v", getter=self.in_s2v)
            obj.set_extension("s2v_vec", getter=self.s2v_vec)
            obj.set_extension("s2v_freq", getter=self.s2v_freq)
            obj.set_extension("s2v_other_senses", getter=self.s2v_other_senses)
            obj.set_extension("s2v_most_similar", method=self.s2v_most_similar)
            obj.set_extension("s2v_similarity", method=self.s2v_similarity)

    def get_phrases(self, doc: Doc) -> List[Span]:
        """Extension attribute getter. Compile a list of sense2vec phrases based
        on a processed Doc: named entities and noun chunks without determiners.

        doc (Doc): The Doc to get phrases from.
        RETURNS (list): The phrases as a list of Span objects.
        """
        func = registry.get_phrases.get(doc._._s2v.cfg["get_phrases"])
        return func(doc)

    def in_s2v(self, obj: Union[Token, Span]) -> bool:
        """Extension attribute getter. Check if a token or span has a vector.

        obj (Token / Span): The object the attribute is called on.
        RETURNS (bool): Whether the key of that object is in the table.
        """
        return self.s2v_key(obj) in obj.doc._._s2v

    def s2v_vec(self, obj: Union[Token, Span]) -> numpy.ndarray:
        """Extension attribute getter. Get the vector for a given object.

        obj (Token / Span): The object the attribute is called on.
        RETURNS (numpy.ndarray): The vector.
        """
        return obj.doc._._s2v[self.s2v_key(obj)]

    def s2v_freq(self, obj: Union[Token, Span]) -> int:
        """Extension attribute getter. Get the frequency for a given object.

        obj (Token / Span): The object the attribute is called on.
        RETURNS (int): The frequency.
        """
        return obj.doc._._s2v.get_freq(self.s2v_key(obj))

    def s2v_key(self, obj: Union[Token, Span]) -> str:
        """Extension attribute getter and helper method. Create a Sense2Vec key
        like "duck|NOUN" from a spaCy object.

        obj (Token / Span): The object to create the key for.
        RETURNS (unicode): The key.
        """
        make_spacy_key = registry.make_spacy_key.get(
            obj.doc._._s2v.cfg["make_spacy_key"]
        )
        if obj.doc._._s2v.cfg.get("lemmatize", False):
            lemma = make_spacy_key(obj, prefer_ents=self.merge_phrases, lemmatize=True)
            lemma_key = obj.doc._._s2v.make_key(*lemma)
            if lemma_key in obj.doc._._s2v:
                return lemma_key
        word, sense = make_spacy_key(obj, prefer_ents=self.merge_phrases)
        return obj.doc._._s2v.make_key(word, sense)

    def s2v_similarity(
        self, obj: Union[Token, Span], other: Union[Token, Span]
    ) -> float:
        """Extension attribute method. Estimate the similarity of two objects.

        obj (Token / Span): The object the attribute is called on.
        other (Token / Span): The object to compare it to.
        RETURNS (float): The similarity score.
        """
        if not isinstance(other, (Token, Span)):
            msg = f"Can only get similarity of Token or Span, not {type(other)}"
            raise ValueError(msg)
        return obj.doc._._s2v.similarity(self.s2v_key(obj), self.s2v_key(other))

    def s2v_most_similar(
        self, obj: Union[Token, Span], n: int = 10
    ) -> List[Tuple[Tuple[str, str], float]]:
        """Extension attribute method. Get the most similar entries.

        obj (Token / Span): The object the attribute is called on.
        n (int): The number of similar entries to return.
        RETURNS (list): The most similar entries as a list of
            ((word, sense), score) tuples.
        """
        key = self.s2v_key(obj)
        results = obj.doc._._s2v.most_similar([key], n=n)
        return [(self.s2v.split_key(result), score) for result, score in results]

    def s2v_other_senses(self, obj: Union[Token, Span]) -> List[str]:
        """Extension attribute getter. Get other senses for an object.

        obj (Token / Span): The object the attribute is called on.
        RETURNS (list): A list of other senses.
        """
        key = self.s2v_key(obj)
        return obj._._s2v.get_other_senses(key)

    def to_bytes(self) -> bytes:
        """Serialize the component to a bytestring.

        RETURNS (bytes): The serialized component.
        """
        return self.s2v.to_bytes(exclude=["strings"])

    def from_bytes(self, bytes_data: bytes):
        """Load the component from a bytestring.

        bytes_data (bytes): The data to load.
        RETURNS (Sense2VecComponent): The loaded object.
        """
        self.s2v = Sense2Vec().from_bytes(bytes_data, exclude=["strings"])
        return self

    def to_disk(self, path: Union[str, Path]):
        """Serialize the component to a directory.

        path (unicode / Path): The path to save to.
        """
        self.s2v.to_disk(path, exclude=["strings"])

    def from_disk(self, path: Union[str, Path]):
        """Load the component from a directory.

        path (unicode / Path): The path to load from.
        RETURNS (Sense2VecComponent): The loaded object.
        """
        self.s2v = Sense2Vec().from_disk(path, exclude=["strings"])
        return self
