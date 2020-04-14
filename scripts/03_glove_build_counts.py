#!/usr/bin/env python
import plac
import os
from pathlib import Path
from wasabi import msg


@plac.annotations(
    glove_dir=("Directory containing the GloVe build", "positional", None, str),
    in_dir=("Directory with preprocessed .s2v files", "positional", None, str),
    out_dir=("Path to output directory", "positional", None, str),
    min_count=("Minimum count for inclusion in vocab", "option", "c", int),
    memory=("Soft limit for memory consumption, in GB", "option", "m", float),
    window_size=("Number of context words on either side", "option", "w", int),
    verbose=("Set verbosity: 0, 1, or 2", "option", "v", int),
)
def main(
    glove_dir, in_dir, out_dir, min_count=5, memory=4.0, window_size=15, verbose=2
):
    """
    Step 3: Build vocabulary and frequency counts

    Expects a directory of preprocessed .s2v input files and will use GloVe to
    collect unigram counts and construct and shuffle cooccurrence data. See here
    for installation instructions: https://github.com/stanfordnlp/GloVe

    Note that this script will call into GloVe and expects you to pass in the
    GloVe build directory (/build if you run the Makefile). The commands will
    also be printed if you want to run them separately.
    """
    input_path = Path(in_dir)
    output_path = Path(out_dir)
    if not Path(glove_dir).exists():
        msg.fail("Can't find GloVe build directory", glove_dir, exits=1)
    if not input_path.exists() or not input_path.is_dir():
        msg.fail("Not a valid input directory", in_dir, exits=1)
    input_files = [str(fp) for fp in input_path.iterdir() if fp.suffix == ".s2v"]
    if not input_files:
        msg.fail("No .s2v files found in input directory", in_dir, exits=1)
    msg.info(f"Using {len(input_files)} input files")
    if not output_path.exists():
        output_path.mkdir(parents=True)
        msg.good(f"Created output directory {out_dir}")

    vocab_file = output_path / f"vocab.txt"
    cooc_file = output_path / f"cooccurrence.bin"
    cooc_shuffle_file = output_path / f"cooccurrence.shuf.bin"

    msg.info("Creating vocabulary counts")
    cmd = (
        f"cat {' '.join(input_files)} | {glove_dir}/vocab_count "
        f"-min-count {min_count} -verbose {verbose} > {vocab_file}"
    )
    print(cmd)
    vocab_cmd = os.system(cmd)
    if vocab_cmd != 0 or not Path(vocab_file).exists():
        msg.fail("Failed creating vocab counts", exits=1)
    msg.good("Created vocab counts", vocab_file)

    msg.info("Creating cooccurrence statistics")
    cmd = (
        f"cat {' '.join(input_files)} | {glove_dir}/cooccur -memory {memory} "
        f"-vocab-file {vocab_file} -verbose {verbose} "
        f"-window-size {window_size} > {cooc_file}"
    )
    print(cmd)
    cooccur_cmd = os.system(cmd)
    if cooccur_cmd != 0 or not Path(cooc_file).exists():
        msg.fail("Failed creating cooccurrence statistics", exits=1)
    msg.good("Created cooccurrence statistics", cooc_file)

    msg.info("Shuffling cooccurrence file")
    cmd = (
        f"{glove_dir}/shuffle -memory {memory} -verbose {verbose} "
        f"< {cooc_file} > {cooc_shuffle_file}"
    )
    print(cmd)
    shuffle_cmd = os.system(cmd)
    if shuffle_cmd != 0 or not Path(cooc_shuffle_file).exists():
        msg.fail("Failed to shuffle cooccurrence file", exits=1)
    msg.good("Shuffled cooccurrence file", cooc_shuffle_file)


if __name__ == "__main__":
    plac.call(main)
