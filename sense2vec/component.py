from typing import Tuple, Union, List
from spacy.tokens import Doc, Token, Span
from spacy.vocab import Vocab
from spacy.language import Language
from pathlib import Path
import numpy

from .sense2vec import Sense2Vec
from .util import merge_phrases, get_phrases, make_spacy_key


class Sense2VecComponent(object):
    name = "sense2vec"

    def __init__(
        self,
        vocab: Vocab = None,
        shape: Tuple[int, int] = (1000, 128),
        merge_phrases: bool = False,
    ):
        strings = vocab.strings if vocab is not None else None
        self.s2v = Sense2Vec(shape=shape, strings=strings)
        self.first_run = True
        self.merge_phrases = merge_phrases

    @classmethod
    def from_nlp(cls, nlp: Language, **kwargs):
        return cls(vocab=nlp.vocab)

    def __call__(self, doc: Doc) -> Doc:
        if self.first_run:
            self.init_component(doc)
            self.first_run = False
        # Store reference to s2v object on Doc to make sure it's right
        doc._._s2v = self.s2v
        if self.merge_phrases:
            doc = merge_phrases(doc)
        return doc

    def init_component(self, doc: Doc):
        # initialise the attributes here only if the component is added to the
        # pipeline and used – otherwise, tokens will still get the attributes
        # even if the component is only created and not added
        Doc.set_extension("_s2v", default=None)
        Doc.set_extension("s2v_phrases", getter=get_phrases)
        Token.set_extension("s2v_key", getter=self.s2v_key)
        Token.set_extension("in_s2v", getter=self.in_s2v)
        Token.set_extension("s2v_vec", getter=self.s2v_vec)
        Token.set_extension("s2v_other_senses", getter=self.s2v_other_senses)
        Token.set_extension("s2v_most_similar", method=self.s2v_most_similar)
        Span.set_extension("s2v_key", getter=self.s2v_key)
        Span.set_extension("in_s2v", getter=self.in_s2v)
        Span.set_extension("s2v_vec", getter=self.s2v_vec)
        Span.set_extension("s2v_other_senses", getter=self.s2v_other_senses)
        Span.set_extension("s2v_most_similar", method=self.s2v_most_similar)

    def get_key(self, obj: Union[Token, Span]) -> str:
        return make_spacy_key(obj, self.s2v.make_key)

    def in_s2v(self, obj: Union[Token, Span]) -> bool:
        return self.get_key(obj) in obj.doc._._s2v

    def s2v_vec(self, obj: Union[Token, Span]) -> numpy.ndarray:
        return obj.doc._._s2v[self.get_key(obj)]

    def s2v_key(self, obj: Union[Token, Span]) -> str:
        return self.get_key(obj)

    def s2v_most_similar(
        self, obj: Union[Token, Span], n_similar: int = 10
    ) -> List[Tuple[Tuple[str, str], float]]:
        key = self.get_key(obj)
        results = obj.doc._._s2v.most_similar([key], n_similar=n_similar)
        return [(self.s2v.split_key(result), score) for result, score in results]

    def s2v_other_senses(self, obj: Union[Token, Span]) -> List[str]:
        key = self.get_key(obj)
        return obj._._s2v.get_other_senses(key)

    def to_bytes(self) -> bytes:
        return self.s2v.to_bytes(exclude=["strings"])

    def from_bytes(self, bytes_data: bytes):
        self.s2v = Sense2Vec().from_bytes(bytes_data, exclude=["strings"])
        return self

    def to_disk(self, path: Union[str, Path]):
        self.s2v.to_disk(path, exclude=["strings"])

    def from_disk(self, path: Union[str, Path]):
        self.s2v = Sense2Vec().from_disk(path, exclude=["strings"])
        return self
