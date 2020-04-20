#!/usr/bin/env python
import plac
from pathlib import Path
from wasabi import msg
import fasttext
from errno import EPIPE
# python 04_fasttext_train_vectors.py /path/to/output/director/ -in /path/to/input/directory


@plac.annotations(
    out_dir=("Path to output directory", "positional", None, str),
    in_dir=("Path to directory with preprocessed .s2v file(s)", "option", "in", str),
    n_threads=("Number of threads", "option", "t", int),
    min_count=("Minimum count for inclusion in vocab", "option", "c", int),
    vector_size=("Dimension of word vector representations", "option", "s", int),
    epoch=("Number of times the fastText model will loop over your data", "option", "e", int),
    save_fasttext_model=("Save fastText model to output directory as a binary file to avoid retraining", "flag", "sv"),
    fasttext_filepath=("Path to saved fastText model .bin file", "option", "ft", str),
    verbose=("Set verbosity: 0, 1, or 2", "option", "v", int),
)
def main(
    out_dir,
    in_dir=None,
    n_threads=10,
    min_count=50,
    vector_size=300,
    epoch=5,
    save_fasttext_model=False,
    fasttext_filepath=None,
    verbose=2,
):
    """
    Step 4: Train the vectors

    Expects a directory of preprocessed .s2v input files, will concatenate them
    (using a temporary file on disk) and will use fastText to train a word2vec
    model. See here for installation instructions:
    https://github.com/facebookresearch/fastText

    Note that this script will call into fastText and expects you to pass in the
    built fasttext binary. The command will also be printed if you want to run
    it separately.
    """

    output_path = Path(out_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        msg.good(f"Created output directory {out_dir}")

    if fasttext_filepath:
        msg.info("Loading fastText model vectors from .bin file")
        if in_dir:
            msg.warn(f"Warning: Providing a fastText filepath overrides fastText vector training")
        fasttext_filepath = Path(fasttext_filepath)
        if not fasttext_filepath.exists() or not fasttext_filepath.is_file() or not (fasttext_filepath.suffix == '.bin'):
            msg.fail("Error: fasttext_filepath expects a fastText model .bin file", exits=1)
        fasttext_model = fasttext.load_model(str(fasttext_filepath))
        msg.good("Successfully loaded fastText model")
    elif in_dir:
        msg.info("Training fastText model vectors")
        input_path = Path(in_dir)
        # Check to see if fasttext_filepath exists
        if not input_path.exists() or not input_path.is_dir():
            msg.fail("Not a valid input directory", in_dir, exits=1)
        tmp_path = input_path / "s2v_input.tmp"
        input_files = [p for p in input_path.iterdir() if p.suffix == ".s2v"]
        if not input_files:
            msg.fail("Input directory contains no .s2v files", in_dir, exits=1)
        # fastText expects only one input file and only reads from disk and not
        # stdin, so we need to create a temporary file that concatenates the inputs
        with tmp_path.open("a", encoding="utf8") as tmp_file:
            for input_file in input_files:
                with input_file.open("r", encoding="utf8") as f:
                    tmp_file.write(f.read())
        msg.info("Created temporary merged input file", tmp_path)
        fasttext_model = fasttext.train_unsupervised(str(tmp_path), thread=n_threads, epoch=epoch, dim=vector_size,
                                                     minn=0, maxn=0, minCount=min_count, verbose=verbose)
        msg.good("Successfully trained fastText model vectors")

        tmp_path.unlink()
        msg.good("Deleted temporary input file", tmp_path)
        output_file = output_path / f"vectors_w2v_{vector_size}dim.bin"
        if save_fasttext_model:
            fasttext_model.save_model(str(output_file))
            if not output_file.exists() or not output_file.is_file():
                msg.fail("Failed to save fastText model to disk", output_file, exits=1)
            msg.good("Successfully saved fastText model to disk", output_file)
    else:
        fasttext_model = None
        msg.fail("Must provide an input directory or fastText binary filepath", exits=1)

    msg.info("Creating vocabulary file")
    vocab_file = output_path / "vocab.txt"
    words, freqs = fasttext_model.get_words(include_freq=True)
    with vocab_file.open('w', encoding='utf8') as f:
        for i in range(len(words)):
            f.write(words[i] + " " + str(freqs[i]) + " word\n")
    if not vocab_file.exists() or not vocab_file.is_file():
        msg.fail("Failed to create vocabulary", vocab_file, exits=1)
    msg.good("Successfully created vocabulary file", vocab_file)

    msg.info("Creating vectors file")
    vectors_file = output_path / "vectors.txt"
    # Adapted from https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/bin_to_vec.py#L31
    with vectors_file.open('w', encoding='utf-8') as file_out:
        # the first line must contain the number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(fasttext_model.get_dimension()) + '\n')
        # line by line, append vector to vectors file
        for w in words:
            v = fasttext_model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except IOError as e:
                if e.errno == EPIPE:
                    pass
    if not vectors_file.exists() or not vectors_file.is_file():
        msg.fail("Failed to create vectors file", vectors_file, exits=1)
    msg.good("Successfully created vectors file", vectors_file)


if __name__ == "__main__":
    plac.call(main)
