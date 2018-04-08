# coding: utf8
from __future__ import unicode_literals

from .vectors import VectorMap
from .about import __version__


def load(vectors_path):
    vector_map = VectorMap(128)
    vector_map.load(vectors_path)
    return vector_map


class Sense2VecComponent(object):
    """
    spaCy v2.0 pipeline component.

    USAGE:
        >>> import spacy
        >>> from sense2vec import Sense2VecComponent
        >>> nlp = spacy.load('en')
        >>> s2v = Sense2VecComponent('/path/to/model')
        >>> nlp.add_pipe(s2v)
        >>> doc = nlp(u"A text about natural language processing.")
        >>> assert doc[3].text == 'natural language processing'
        >>> assert doc[3]._.in_s2v
        >>> print(doc[3]._.s2v_most_similar(20))
    """
    name = 'sense2vec'

    def __init__(self, vectors_path):
        self.s2v = load(vectors_path)
        self.first_run = True

    def __call__(self, doc):
        if self.first_run:
            self.init_component(doc)
            self.first_run = False
        if not doc.is_tagged:
            raise ValueError("Can't run sense2vec: document not tagged.")
        for ent in doc.ents:
            ent.merge(tag=ent.root.tag_, lemma=ent.root.lemma_,
                      ent_type=ent.label_)
        for np in doc.noun_chunks:
            while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
                np = np[1:]
            np.merge(tag=np.root.tag_, lemma=np.root.lemma_,
                     ent_type=np.root.ent_type_)
        return doc

    def init_component(self, doc):
        # initialise the attributes here only if the component is added to the
        # pipeline and used – otherwise, tokens will still get the attributes
        # even if the component is only created and not added
        Token = doc[0].__class__
        Span = doc[:1].__class__
        Token.set_extension('in_s2v', getter=lambda t: self.in_s2v(t))
        Token.set_extension('s2v_freq', getter=lambda t: self.s2v_freq(t))
        Token.set_extension('s2v_vec', getter=lambda t: self.s2v_vec(t))
        Token.set_extension('s2v_most_similar', method=lambda t, n: self.s2v_most_sim(t, n))
        Span.set_extension('in_s2v', getter=lambda s: self.in_s2v(s, 'ent'))
        Span.set_extension('s2v_freq', getter=lambda s: self.s2v_freq(s, 'ent'))
        Span.set_extension('s2v_vec', getter=lambda s: self.s2v_vec(s, 'ent'))
        Span.set_extension('s2v_most_similar', method=lambda s, n: self.s2v_most_sim(s, n, 'ent'))

    def in_s2v(self, obj, attr='pos'):
        return self._get_query(obj, attr) in self.s2v

    def s2v_freq(self, obj, attr='pos'):
        freq, _ = self.s2v[self._get_query(obj, attr)]
        return freq

    def s2v_vec(self, obj, attr='pos'):
        _, vector = self.s2v[self._get_query(obj, attr)]
        return vector

    def s2v_most_sim(self, obj, n_similar=10, attr='pos'):
        _, vector = self.s2v[self._get_query(obj, attr)]
        words, scores = self.s2v.most_similar(vector, n_similar)
        words = [word.replace('_', ' ') for word in words]
        words = [tuple(word.rsplit('|', 1)) for word in words]
        return list(zip(words, scores))

    def _get_query(self, obj, attr='pos'):
        # no pos_ and label_ shouldn't happen – unless it's an unmerged
        # non-entity Span (in which case we just use the root's pos)
        pos = obj.pos_ if hasattr(obj, 'pos_') else obj.root.pos_
        sense = obj.label_ if (attr == 'ent' and obj.label_) else pos
        return obj.text.replace(' ', '_') + '|' + sense
