#!/usr/bin/env python
import os
from pathlib import Path
from wasabi import msg
import typer


def main(
    # fmt: off
    glove_dir: str = typer.Argument(..., help="Directory containing the GloVe build"),
    in_file: str = typer.Argument(..., help="Input file (shuffled cooccurrences)"),
    vocab_file: str = typer.Argument(..., help="Vocabulary file"),
    out_dir: str = typer.Argument(..., help="Path to output directory"),
    n_threads: int = typer.Option(8, "--n-threads", "-t", help="Number of threads"),
    n_iter: int = typer.Option(15, "--n-iter", "-n", help="Number of iterations"),
    x_max: int = typer.Option(10, "--x-max", "-x", help="Parameter specifying cutoff in weighting function"),
    vector_size: int = typer.Option(128, "--vector-size", "-s", help="Dimension of word vector representations"),
    verbose: int = typer.Option(2, "--verbose", "-v", help="Set verbosity: 0, 1, or 2"),
    # fmt: on
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
    typer.run(main)
