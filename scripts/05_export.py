#!/usr/bin/env python
from sense2vec import Sense2Vec
from sense2vec.util import split_key
from pathlib import Path
import plac
from wasabi import msg
import numpy


def _get_shape(file_):
    """Return a tuple with (number of entries, vector dimensions). Handle
    both word2vec/FastText format, which has a header with this, or GloVe's
    format, which doesn't."""
    first_line = next(file_).split()
    if len(first_line) == 2:
        return tuple(int(size) for size in first_line), file_
    count = 1
    for line in file_:
        count += 1
    file_.seek(0)
    shape = (count, len(first_line) - 1)
    return shape, file_


@plac.annotations(
    in_file=("Vectors file (text-based)", "positional", None, str),
    vocab_file=("Vocabulary file", "positional", None, str),
    out_dir=("Path to output directory", "positional", None, str),
)
def main(in_file, vocab_file, out_dir):
    """
    Step 5: Export a sense2vec component

    Expects a vectors.txt and a vocab file trained with GloVe and exports
    a component that can be loaded with Sense2vec.from_disk.
    """
    input_path = Path(in_file)
    vocab_path = Path(vocab_file)
    output_path = Path(out_dir)
    if not input_path.exists():
        msg.fail("Can't find input file", in_file, exits=1)
    if input_path.suffix == ".bin":
        msg.fail("Need text-based vectors file, not binary", in_file, exits=1)
    if not vocab_path.exists():
        msg.fail("Can't find vocab file", vocab_file, exits=1)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        msg.good(f"Created output directory {out_dir}")
    with input_path.open("r", encoding="utf8") as f:
        (n_vectors, vector_size), f = _get_shape(f)
        vectors_data = f.readlines()
    with vocab_path.open("r", encoding="utf8") as f:
        vocab_data = f.readlines()
    data = []
    all_senses = set()
    for item in vectors_data:
        item = item.rstrip().rsplit(" ", vector_size)
        key = item[0]
        try:
            _, sense = split_key(key)
        except ValueError:
            continue
        vec = item[1:]
        if len(vec) != vector_size:
            msg.fail(f"Wrong vector size: {len(vec)} (expected {vector_size})", exits=1)
        all_senses.add(sense)
        data.append((key, numpy.asarray(vec, dtype=numpy.float32)))
    s2v = Sense2Vec(shape=(len(data), vector_size), senses=all_senses)
    for key, vector in data:
        s2v.add(key, vector)
    for item in vocab_data:
        item = item.rstrip()
        if item.endswith(" word"):  # for fastText vocabs
            item = item[:-5]
        try:
            key, freq = item.rsplit(" ", 1)
        except ValueError:
            continue
        s2v.set_freq(key, int(freq))
    msg.good("Created the sense2vec model")
    msg.info(f"{len(data)} vectors, {len(all_senses)} total senses")
    s2v.to_disk(output_path)
    msg.good("Saved model to directory", out_dir)


if __name__ == "__main__":
    plac.call(main)
