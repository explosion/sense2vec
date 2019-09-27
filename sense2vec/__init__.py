# coding: utf8
from __future__ import unicode_literals

from pathlib import Path
from spacy.vectors import Vectors
from spacy.strings import StringStore
from spacy.tokens import Doc, Token, Span
import numpy
import srsly

from .util import transform_doc, get_phrases, make_key, split_key
from .about import __version__  # noqa: F401


def load(vectors_path):
    vectors_path = Path(vectors_path)
    if not vectors_path.exists():
        raise IOError("Can't find vectors: {}".format(vectors_path))
    return Sense2Vec().from_disk(vectors_path)


class Sense2Vec(object):
    def __init__(self, shape=(1000, 128), strings=None):
        self.vectors = Vectors(shape=shape)
        self.strings = StringStore() if strings is None else strings

    def __len__(self):
        return len(self.vectors)

    def __contains__(self, key):
        key = key if isinstance(key, int) else self.strings[key]
        return key in self.vectors

    def __getitem__(self, key):
        key = key if isinstance(key, int) else self.strings[key]
        if key in self.vectors:
            return self.vectors[key]

    def __iter__(self):
        yield from self.items()

    def add(self, key, vector):
        if not isinstance(key, int):
            key = self.strings.add(key)
        self.vectors.add(key, vector=vector)

    def items(self):
        for key, value in self.vectors.items():
            yield self.strings[key], value

    def keys(self):
        for key in self.vectors.keys():
            yield self.strings[key]

    def values(self):
        yield from self.vectors.values()

    def most_similar(self, keys, n_similar=10):
        if not isinstance(keys, (list, tuple)):
            raise ValueError("Expected iterable of keys. Got: {}".format(type(keys)))
        vecs = [self[key] for key in keys if key in self]
        queries = numpy.asarray(vecs, dtype=numpy.float32)
        result_keys, _, scores = self.vectors.most_similar(queries)
        result = zip(result_keys, scores)
        result = [(self.strings[key], score) for key, score in result if key]
        result = [(key, score) for key, score in result if key not in keys]
        # TODO: handle this better?
        return result[:n_similar]

    def to_bytes(self, exclude=tuple()):
        data = {"vectors": self.vectors.to_bytes()}
        if "strings" not in exclude:
            data["strings"] = self.strings.to_bytes()
        return srsly.msgpack_dumps(data)

    def from_bytes(self, bytes_data, exclude=tuple()):
        data = srsly.msgpack_loads(bytes_data)
        self.vectors = Vectors().from_bytes(data["vectors"])
        if "strings" not in exclude and "strings" in data:
            self.strings = StringStore().from_bytes(data["strings"])
        return self

    def from_disk(self, path, exclude=tuple()):
        path = Path(path)
        strings_path = path / "strings.json"
        self.vectors = Vectors().from_disk(path)
        if "strings" not in exclude and strings_path.exists():
            self.strings = StringStore().from_disk(strings_path)
        return self

    def to_disk(self, path, exclude=tuple()):
        path = Path(path)
        self.vectors.to_disk(path)
        if "strings" not in exclude:
            self.strings.to_disk(path / "strings.json")
        return self


class Sense2VecComponent(object):
    name = "sense2vec"

    def __init__(
        self,
        vocab=None,
        shape=(1000, 128),
        merge_phrases=False,
        make_key=make_key,
        split_key=split_key,
    ):
        strings = vocab.strings if vocab is not None else None
        self.s2v = Sense2Vec(shape=shape, strings=strings)
        self.first_run = True
        self.merge_phrases = merge_phrases
        self.make_key = make_key
        self.split_key = split_key

    @classmethod
    def from_nlp(cls, nlp, **kwargs):
        return cls(vocab=nlp.vocab)

    def __call__(self, doc):
        if self.first_run:
            self.init_component(doc)
            self.first_run = False
        # Store reference to s2v object on Doc to make sure it's right
        doc._._s2v = self.s2v
        if self.merge_phrases:
            doc = transform_doc(doc)
        return doc

    def init_component(self, doc):
        # initialise the attributes here only if the component is added to the
        # pipeline and used – otherwise, tokens will still get the attributes
        # even if the component is only created and not added
        Doc.set_extension("_s2v", default=None)
        Doc.set_extension("s2v_phrases", getter=get_phrases)
        Token.set_extension("s2v_key", getter=self.s2v_key)
        Token.set_extension("in_s2v", getter=self.in_s2v)
        Token.set_extension("s2v_vec", getter=self.s2v_vec)
        Token.set_extension("s2v_most_similar", method=self.s2v_most_sim)
        Span.set_extension("s2v_key", getter=self.s2v_key)
        Span.set_extension("in_s2v", getter=self.in_s2v)
        Span.set_extension("s2v_vec", getter=self.s2v_vec)
        Span.set_extension("s2v_most_similar", method=self.s2v_most_sim)

    def in_s2v(self, obj):
        return self.make_key(obj) in obj.doc._._s2v

    def s2v_vec(self, obj):
        return obj.doc._._s2v[self.make_key(obj)]

    def s2v_key(self, obj):
        return self.make_key(obj)

    def s2v_most_sim(self, obj, n_similar=10):
        key = self.make_key(obj)
        results = obj.doc._._s2v.most_similar([key], n_similar=n_similar)
        return [(self.split_key(result), score) for result, score in results]

    def to_bytes(self):
        return self.s2v.to_bytes(exclude=["strings"])

    def from_bytes(self, bytes_data):
        self.s2v = Sense2Vec().from_bytes(bytes_data, exclude=["strings"])
        return self

    def to_disk(self, path):
        self.s2v.to_bytes(path, exclude=["strings"])

    def from_disk(self, path):
        self.s2v = Sense2Vec().from_disk(path, exclude=["strings"])
        return self
