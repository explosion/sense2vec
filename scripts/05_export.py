#!/usr/bin/env python
from collections import OrderedDict, defaultdict
from sense2vec import Sense2Vec
from sense2vec.util import split_key, cosine_similarity
from pathlib import Path
from wasabi import msg
import numpy
import typer


def main(
    # fmt: off
    in_file: str = typer.Argument(..., help="Vectors file (text-based)"),
    vocab_file: str = typer.Argument(..., help="Vocabulary file"),
    out_dir: str = typer.Argument(..., help="Path to output directory"),
    min_freq_ratio: float = typer.Option(0.0, "--min-freq-ratio", "-r", help="Frequency ratio threshold for discarding minority senses or casings"),
    min_distance: float = typer.Option(0.0, "--min-distance", "-s", help="Similarity threshold for discarding redundant keys"),
    # fmt: on
):
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
        vocab = read_vocab(f)
    vectors = {}
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
        vectors[key] = numpy.asarray(vec, dtype=numpy.float32)
    discarded = set()
    discarded.update(get_minority_keys(vocab, min_freq_ratio))
    discarded.update(get_redundant_keys(vocab, vectors, min_distance))
    n_vectors = len(vectors) - len(discarded)
    s2v = Sense2Vec(shape=(n_vectors, vector_size), senses=list(all_senses))
    for key, vector in vectors.items():
        if key not in discarded:
            s2v.add(key, vector)
            s2v.set_freq(key, vocab[key])
    msg.good("Created the sense2vec model")
    msg.info(f"{n_vectors} vectors, {len(all_senses)} total senses")
    s2v.to_disk(output_path)
    msg.good("Saved model to directory", out_dir)


def _get_shape(file_):
    """Return a tuple with (number of entries, vector dimensions). Handle
    both word2vec/FastText format, which has a header with this, or GloVe's
    format, which doesn't."""
    first_line = next(file_).replace("\ufeff", "").split()
    if len(first_line) == 2:
        return tuple(int(size) for size in first_line), file_
    count = 1
    for line in file_:
        count += 1
    file_.seek(0)
    shape = (count, len(first_line) - 1)
    return shape, file_


def read_vocab(vocab_file):
    freqs = OrderedDict()
    for line in vocab_file:
        item = line.rstrip()
        if item.endswith(" word"):  # for fastText vocabs
            item = item[:-5]
        try:
            key, freq = item.rsplit(" ", 1)
        except ValueError:
            continue
        freqs[key] = int(freq)
    return freqs


def get_minority_keys(freqs, min_ratio):
    """Remove keys that are too infrequent relative to a main sense."""
    by_word = defaultdict(list)
    for key, freq in freqs.items():
        try:
            term, sense = split_key(key)
        except ValueError:
            continue
        if freq:
            by_word[term.lower()].append((freq, key))
    discarded = []
    for values in by_word.values():
        if len(values) >= 2:
            values.sort(reverse=True)
            freq1, key1 = values[0]
            for freq2, key2 in values[1:]:
                ratio = freq2 / freq1
                if ratio < min_ratio:
                    discarded.append(key2)
    return discarded


def get_redundant_keys(vocab, vectors, min_distance):
    if min_distance <= 0.0:
        return []
    by_word = defaultdict(list)
    for key, freq in vocab.items():
        try:
            term, sense = split_key(key)
        except ValueError:
            continue
        term = term.split("_")[-1]
        by_word[term.lower()].append((freq, key))
    too_similar = []
    for values in by_word.values():
        if len(values) >= 2:
            values.sort(reverse=True)
            freq1, key1 = values[0]
            vector1 = vectors[key1]
            for freq2, key2 in values[1:]:
                vector2 = vectors[key2]
                sim = cosine_similarity(vector1, vector2)
                if sim >= (1 - min_distance):
                    too_similar.append(key2)
    return too_similar


if __name__ == "__main__":
    typer.run(main)
