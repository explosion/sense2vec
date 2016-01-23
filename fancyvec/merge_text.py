from __future__ import print_function, unicode_literals, division
import io
import bz2
import logging

import spacy.en
from preshed.counter import PreshCounter

import joblib
import plac
import ujson
from gensim.models import Word2Vec


def parallelize(func, iterator, n_jobs, extra_args):
    Parallel(n_jobs=n_jobs)(delayed(func)(*(item + extra_args))
             for item in iterator)


def iter_comments(loc):
    with bz2.BZ2File(self.loc) as file_:
        for line in file_:
            yield ujson.loads(line)['body']


def transform_texts(input_):
    nlp = English()
    output = []
    for text in input_:
        doc = self.nlp(text)
        for np in doc.noun_chunks:
            while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
                np = np[1:]
            if len(np):
                np.merge(np.root.tag_, np.text, np.root.ent_type_)
        for ent in doc.ents:
            if len(ent) > 1:
                ent.merge(ent.root.tag_, ent.text + '|' + ent.label_, ent.label_)
        output.append([word.text.replace(' ', '_') for word in doc])
    return output


class SpacyCorpus(object):
    def __init__(self, loc):
        self.loc = loc

    def __iter__(self):
        texts = read_comments(self.loc)
        results = parallelize(transform_texts, texts, self.n_workers)
        
        for batch in results:
            for text in batch:
                yield text


@plac.annotations(
    loc=("Location of input file"),
    n_workers=("Number of workers", "option", "n", int),
    size=("Dimension of the word vectors", "option", "d", int),
    window=("Context window size", "option", "w", int),
    min_count=("Min count", "option", "m", int)
)
def main(loc, n_workers=8, window=5, size=10, min_count=10):
    nlp = spacy.en.English()
    corpus = SpacyCorpus(nlp, loc)

    model = Word2Vec(
        corpus,
        size=size,
        window=window,
        min_count=min_count,
        workers=n_workers
    )

    for word in model.vocab:
        count = model.vocab[word].count
        print(word, count)


if __name__ == '__main__':
    plac.call(main)
