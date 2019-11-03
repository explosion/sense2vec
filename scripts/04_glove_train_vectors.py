#!/usr/bin/env python
import plac
import os
from pathlib import Path
from wasabi import msg


@plac.annotations(
    glove_dir=("Directory containing the GloVe build", "positional", None, str),
    in_file=("Input file (shuffled cooccurrences)", "positional", None, str),
    vocab_file=("Vocabulary file", "positional", None, str),
    out_dir=("Path to output directory", "positional", None, str),
    n_threads=("Number of threads", "option", "t", int),
    n_iter=("Number of iterations", "option", "n", int),
    x_max=("Parameter specifying cutoff in weighting function", "option", "x", int),
    vector_size=("Dimension of word vector representations", "option", "s", int),
    verbose=("Set verbosity: 0, 1, or 2", "option", "v", int),
)
def main(
    glove_dir,
    in_file,
    vocab_file,
    out_dir,
    n_threads=8,
    n_iter=15,
    x_max=10,
    vector_size=128,
    verbose=2,
):
    """
    Step 4: Train the vectors

    Expects a file containing the shuffled cooccurrences and a vocab file and
    will output a plain-text vectors file.

    Note that this script will call into GloVe and expects you to pass in the
    GloVe build directory (/build if you run the Makefile). The commands will
    also be printed if you want to run them separately.
    """
    output_path = Path(out_dir)
    if not Path(glove_dir).exists():
        msg.fail("Can't find GloVe build directory", glove_dir, exits=1)
    if not Path(in_file).exists():
        msg.fail("Can't find input file", in_file, exits=1)
    if not Path(vocab_file).exists():
        msg.fail("Can't find vocab file", vocab_file, exits=1)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        msg.good(f"Created output directory {out_dir}")
    output_file = output_path / f"vectors_glove_{vector_size}dim"
    msg.info("Training vectors")
    cmd = (
        f"{glove_dir}/glove -save-file {output_file} -threads {n_threads} "
        f"-input-file {in_file} -x-max {x_max} -iter {n_iter} "
        f"-vector-size {vector_size} -binary 0 -vocab-file {vocab_file} "
        f"-verbose {verbose}"
    )
    print(cmd)
    train_cmd = os.system(cmd)
    if train_cmd != 0:
        msg.fail("Failed training vectors", exits=1)
    msg.good("Successfully trained vectors")


if __name__ == "__main__":
    plac.call(main)
