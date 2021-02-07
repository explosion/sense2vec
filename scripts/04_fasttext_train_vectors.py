#!/usr/bin/env python
from typing import Optional
from pathlib import Path
from wasabi import msg
import fasttext
from errno import EPIPE
import typer

# python 04_fasttext_train_vectors.py /path/to/output/director/ -in /path/to/input/directory


def main(
    # fmt: off
    out_dir: str = typer.Argument(..., help="Path to output directory"),
    in_dir: Optional[str] = typer.Argument(None, help="Path to directory with preprocessed .s2v file(s)"),
    n_threads: int = typer.Option(10, "--n-threads", "-t", help="Number of threads"),
    min_count: int = typer.Option(50, "--min-count", "-c", help="Minimum count for inclusion in vocab"),
    vector_size: int = typer.Option(300, "--vector-size", "-s", help="Dimension of word vector representations"),
    epoch: int = typer.Option(5, "--epoch", "-e", help="Number of times the fastText model will loop over your data"),
    save_fasttext_model: bool = typer.Option(False, "--save-fasttext-model", "-sv", help="Save fastText model to output directory as a binary file to avoid retraining"),
    fasttext_filepath: Optional[str] = typer.Option(None, "--fasttext-filepath", "-ft", help="Path to saved fastText model .bin file"),
    verbose: int = typer.Option(2, "--verbose", "-v", help="Set verbosity: 0, 1, or 2"),
    # fmt: on
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
            msg.warn(
                f"Warning: Providing a fastText filepath overrides fastText vector training"
            )
        fasttext_filepath = Path(fasttext_filepath)
        if (
            not fasttext_filepath.exists()
            or not fasttext_filepath.is_file()
            or not (fasttext_filepath.suffix == ".bin")
        ):
            msg.fail(
                "Error: fasttext_filepath expects a fastText model .bin file", exits=1
            )
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
        fasttext_model = fasttext.train_unsupervised(
            str(tmp_path),
            thread=n_threads,
            epoch=epoch,
            dim=vector_size,
            minn=0,
            maxn=0,
            minCount=min_count,
            verbose=verbose,
        )
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
    with vocab_file.open("w", encoding="utf8") as f:
        for i in range(len(words)):
            f.write(words[i] + " " + str(freqs[i]) + " word\n")
    if not vocab_file.exists() or not vocab_file.is_file():
        msg.fail("Failed to create vocabulary", vocab_file, exits=1)
    msg.good("Successfully created vocabulary file", vocab_file)

    msg.info("Creating vectors file")
    vectors_file = output_path / "vectors.txt"
    # Adapted from https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/bin_to_vec.py#L31
    with vectors_file.open("w", encoding="utf-8") as file_out:
        # the first line must contain the number of total words and vector dimension
        file_out.write(
            str(len(words)) + " " + str(fasttext_model.get_dimension()) + "\n"
        )
        # line by line, append vector to vectors file
        for w in words:
            v = fasttext_model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + "\n")
            except IOError as e:
                if e.errno == EPIPE:
                    pass
    if not vectors_file.exists() or not vectors_file.is_file():
        msg.fail("Failed to create vectors file", vectors_file, exits=1)
    msg.good("Successfully created vectors file", vectors_file)


if __name__ == "__main__":
    typer.run(main)
