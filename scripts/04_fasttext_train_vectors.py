#!/usr/bin/env python
import plac
import os
from pathlib import Path
from wasabi import msg


@plac.annotations(
    fasttext_bin=("Path to the fasttext binary", "positional", None, str),
    in_dir=("Directory with preprocessed .s2v files", "positional", None, str),
    out_dir=("Path to output directory", "positional", None, str),
    n_threads=("Number of threads", "option", "t", int),
    min_count=("Minimum count for inclusion in vocab", "option", "c", int),
    vector_size=("Dimension of word vector representations", "option", "s", int),
    verbose=("Set verbosity: 0, 1, or 2", "option", "v", int),
)
def main(
    fasttext_bin,
    in_dir,
    out_dir,
    n_threads=10,
    min_count=50,
    vector_size=300,
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
    input_path = Path(in_dir)
    output_path = Path(out_dir)
    if not Path(fasttext_bin).exists():
        msg.fail("Can't find fastText binary", fasttext_bin, exits=1)
    if not input_path.exists() or not input_path.is_dir():
        msg.fail("Not a valid input directory", in_dir, exits=1)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        msg.good(f"Created output directory {out_dir}")
    output_file = output_path / f"vectors_w2v_{vector_size}dim"
    # fastText expects only one input file and only reads from disk and not
    # stdin, so we need to create a temporary file that concatenates the inputs
    tmp_path = input_path / "s2v_input.tmp"
    input_files = [p for p in input_path.iterdir() if p.suffix == ".s2v"]
    if not input_files:
        msg.fail("Input directory contains no .s2v files", in_dir, exits=1)
    with tmp_path.open("a", encoding="utf8") as tmp_file:
        for input_file in input_files:
            with input_file.open("r", encoding="utf8") as f:
                tmp_file.write(f.read())
    msg.info("Created temporary merged input file", tmp_path)

    msg.info("Training vectors")
    cmd = (
        f"{fasttext_bin} skipgram -thread {n_threads} -input {tmp_path} "
        f"-output {output_file} -dim {vector_size} -minn 0 -maxn 0 "
        f"-minCount {min_count} -verbose {verbose}"
    )
    print(cmd)
    train_cmd = os.system(cmd)
    tmp_path.unlink()
    msg.good("Deleted temporary input file", tmp_path)
    if train_cmd != 0:
        msg.fail("Failed training vectors", exits=1)
    msg.good("Successfully trained vectors", out_dir)

    msg.info("Creating vocabulary")
    vocab_file = output_path / "vocab.txt"
    cmd = f"{fasttext_bin} dump {output_file.with_suffix('.bin')} dict > {vocab_file}"
    print(cmd)
    vocab_cmd = os.system(cmd)
    if vocab_cmd != 0:
        msg.fail("Failed creating vocabulary", exits=1)
    msg.good("Successfully created vocabulary file", vocab_file)


if __name__ == "__main__":
    plac.call(main)
