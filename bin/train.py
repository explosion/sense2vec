#!/usr/bin/env python
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from sense2vec import Sense2Vec
from sense2vec.util import split_key
import plac


@plac.annotations(
    in_dir=("Location of input directory", "positional", None, str),
    out_file=("Location of output file", "positional", None, str),
    n_workers=("Number of workers", "option", "n", int),
    size=("Dimension of the word vectors", "option", "d", int),
    window=("Context window size", "option", "w", int),
    min_count=("Min count", "option", "m", int),
    negative=("Number of negative samples", "option", "g", int),
    nr_iter=("Number of iterations", "option", "i", int),
)
def main(
    in_dir,
    out_file,
    negative=5,
    n_workers=4,
    window=5,
    size=128,
    min_count=10,
    nr_iter=2,
):
    w2v_model = Word2Vec(
        size=size,
        window=window,
        min_count=min_count,
        workers=n_workers,
        sample=1e-5,
        negative=negative,
        iter=nr_iter,
    )
    sentences = PathLineSentences(in_dir)
    print("Building the vocabulary...")
    w2v_model.build_vocab(sentences)
    print("Training the model...")
    w2v_model.train(
        sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter
    )
    print("Creating the sense2vec model...")
    vectors = []
    all_senses = set()
    for string in w2v_model.wv.vocab:
        vocab = w2v_model.wv.vocab[string]
        freq, idx = vocab.count, vocab.index
        if freq < min_count:
            continue
        vector = w2v_model.wv.vectors[idx]
        vectors.append((string, freq, vector))
        _, sense = split_key(string)
        all_senses.add(sense)
    s2v = Sense2Vec(shape=(len(vectors), size), senses=all_senses)
    for string, _, vector in vectors:
        s2v.add(string, vector)
    print("Saving the model...")
    s2v.to_disk(out_file)
    print(f"Saved model to file: {out_file}")


if __name__ == "__main__":
    plac.call(main)
