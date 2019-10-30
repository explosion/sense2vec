#!/usr/bin/env python
from sense2vec.util import merge_phrases, make_spacy_key
import spacy
from pathlib import Path
import plac
import tqdm
from wasabi import Printer


def represent_doc(doc):
    strings = []
    for sent in doc.sents:
        if sent.text.strip():
            words = " ".join(
                make_spacy_key(w, prefer_ents=True) for w in sent if not w.is_space
            )
            strings.append(words)
    return "\n".join(strings) + "\n" if strings else ""


@plac.annotations(
    in_file=("Path to input file", "positional", None, str),
    out_file=("Path to output file", "positional", None, str),
    spacy_model=("Name of spaCy model to use", "positional", None, str),
    n_process=("Number of processes (multiprocessing)", "option", "n", int),
)
def main(in_file, out_file, spacy_model="en_core_web_sm", n_process=1):
    """
    This script can be used to preprocess a corpus for training a sense2vec
    model. It takes a text file with one sentence per line, and outputs a text
    file with one sentence per line in the expected sense2vec format (merged
    noun phrases, concatenated phrases with underscores and added "senses").

    Example input:
    Rats, mould and broken furniture: the scandal of the UK's refugee housing

    Example output:
    Rats|NOUN ,|PUNCT mould|NOUN and|CCONJ broken_furniture|NOUN :|PUNCT
    the|DET scandal|NOUN of|ADP the|DET UK|GPE 's|PART refugee_housing|NOUN
    """
    msg = Printer()
    input_path = Path(in_file)
    output_path = Path(out_file)
    if not input_path.exists():
        msg.fail("Can't find input file", in_file, exits=1)
    nlp = spacy.load(spacy_model)
    msg.info(f"Using spaCy model {spacy_model}")
    nlp.add_pipe(merge_phrases, name="merge_sense2vec_phrases")
    lines_count = 0
    msg.text("Preprocessing text...")
    with input_path.open("r", encoding="utf8") as texts:
        docs = nlp.pipe(texts, n_process=n_process)
        lines = (represent_doc(doc) for doc in docs)
        with output_path.open("w", encoding="utf8") as f:
            for line in tqdm.tqdm(lines, desc="Lines", unit=""):
                lines_count += 1
                f.write(line)
    msg.good(f"Successfully preprocessed {lines_count} lines", output_path.resolve())


if __name__ == "__main__":
    plac.call(main)
