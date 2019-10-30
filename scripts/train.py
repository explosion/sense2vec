#!/usr/bin/env python
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from sense2vec import Sense2Vec
from sense2vec.util import split_key
from pathlib import Path
import plac
import logging
from wasabi import Printer


@plac.annotations(
    input_data=("Location of input directory or text file", "positional", None, str),
    output_dir=("Location of output directory", "positional", None, str),
    n_workers=("Number of workers", "option", "n", int),
    size=("Dimension of the vectors", "option", "s", int),
    window=("Context window size", "option", "w", int),
    min_count=("Minimum frequency of terms to be included", "option", "m", int),
    negative=("Number of negative examples for Word2Vec", "option", "g", int),
    n_iter=("Number of iterations", "option", "i", int),
    verbose=("Log debugging info", "flag", "V", bool),
)
def main(
    input_data,
    output_dir,
    negative=5,
    n_workers=4,
    window=5,
    size=128,
    min_count=10,
    n_iter=2,
    verbose=False,
):
    """Train a sense2vec model using Gensim. Accepts a text file or a directory
    of text files in the format created by the preprocessing script. Saves out
    a sense2vec model component that can be loaded via Sense2Vec.from_disk.
    """
    msg = Printer(hide_animation=verbose)
    if not Path(input_data).exists():
        msg.fail("Can't find input data (file or directory)", input_data, exits=1)
    if verbose:
        logging.basicConfig(
            format="%(asctime)s - %(message)s", datefmt="%H:%M:%S", level=logging.INFO
        )
    w2v_model = Word2Vec(
        size=size,
        window=window,
        min_count=min_count,
        workers=n_workers,
        sample=1e-5,
        negative=negative,
        iter=n_iter,
    )
    sentences = PathLineSentences(input_data)
    msg.info(f"Using input data from {len(sentences.input_files)} file(s)")
    with msg.loading("Building the vocabulary..."):
        w2v_model.build_vocab(sentences)
    msg.good("Built the vocabulary")
    with msg.loading("Training the model..."):
        w2v_model.train(
            sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter
        )
    msg.good("Trained the model")
    vectors = []
    all_senses = set()
    with msg.loading("Creating the sense2vec model..."):
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
        for string, freq, vector in vectors:
            s2v.add(string, vector, freq)
    msg.good("Created the sense2vec model")
    msg.info(f"{len(vectors)} vectors, {len(all_senses)} total senses")
    with msg.loading("Saving the model..."):
        output_path = Path(output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True)
        s2v.to_disk(output_path)
    msg.good("Saved model to directory", output_dir)


if __name__ == "__main__":
    plac.call(main)
