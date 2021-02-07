#!/usr/bin/env python
from sense2vec.util import make_key, make_spacy_key, merge_phrases
import spacy
from spacy.tokens import DocBin
from wasabi import msg
from pathlib import Path
import tqdm
import typer


def main(
    # fmt: off
    in_file: str = typer.Argument(..., help="Path to input file"),
    out_dir: str = typer.Argument(..., help="Path to output directory"),
    spacy_model: str = typer.Argument("en_core_web_sm", help="Name of spaCy model to use"),
    n_process: int = typer.Option(1, "--n-process", "-n", help="Number of processes (multiprocessing)"),
    # fmt: on
):
    """
    Step 2: Preprocess text in sense2vec's format

    Expects a binary .spacy input file consisting of the parsed Docs (DocBin)
    and outputs a text file with one sentence per line in the expected sense2vec
    format (merged noun phrases, concatenated phrases with underscores and
    added "senses").

    Example input:
    Rats, mould and broken furniture: the scandal of the UK's refugee housing

    Example output:
    Rats|NOUN ,|PUNCT mould|NOUN and|CCONJ broken_furniture|NOUN :|PUNCT
    the|DET scandal|NOUN of|ADP the|DET UK|GPE 's|PART refugee_housing|NOUN
    """
    input_path = Path(in_file)
    output_path = Path(out_dir)
    if not input_path.exists():
        msg.fail("Can't find input file", in_file, exits=1)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        msg.good(f"Created output directory {out_dir}")
    nlp = spacy.load(spacy_model)
    msg.info(f"Using spaCy model {spacy_model}")
    with input_path.open("rb") as f:
        doc_bin_bytes = f.read()
    doc_bin = DocBin().from_bytes(doc_bin_bytes)
    msg.good(f"Loaded {len(doc_bin)} parsed docs")
    docs = doc_bin.get_docs(nlp.vocab)
    output_file = output_path / f"{input_path.stem}.s2v"
    lines_count = 0
    words_count = 0
    with output_file.open("w", encoding="utf8") as f:
        for doc in tqdm.tqdm(docs, desc="Docs", unit=""):
            doc = merge_phrases(doc)
            words = []
            for token in doc:
                if not token.is_space:
                    word, sense = make_spacy_key(token, prefer_ents=True)
                    words.append(make_key(word, sense))
            f.write(" ".join(words) + "\n")
            lines_count += 1
            words_count += len(words)
    msg.good(
        f"Successfully preprocessed {lines_count} docs ({words_count} words)",
        output_file.resolve(),
    )


if __name__ == "__main__":
    typer.run(main)
