#!/usr/bin/env python
# coding: utf-8
"""This script can be used to preprocess a corpus for training a sense2vec
model. It take text file with one sentence per line, and outputs a text file
with one sentence per line in the expected sense2vec format (merged noun
phrases, concatenated phrases with underscores and added "senses").

Example input:
Rats, mould and broken furniture: the scandal of the UK's refugee housing

Example output:
Rats|NOUN ,|PUNCT mould|NOUN and|CCONJ broken_furniture|NOUN :|PUNCT
the|DET scandal|NOUN of|ADP the|DET UK|GPE 's|PART refugee_housing|NOUN

DISCLAIMER: The sense2vec training and preprocessing tools are still a work in
progress. Please note that this script hasn't been optimised for efficiency yet
and doesn't paralellize or batch up any of the work, so you might have to
add this functionality yourself for now.
"""
from __future__ import print_function, unicode_literals

from sense2vec import transform_doc
import spacy
from pathlib import Path
from tqdm import tqdm
import re
import plac


def represent_word(word):
    if word.like_url:
        return '%%URL|X'
    text = re.sub(r'\s', '_', word.text)
    tag = word.ent_type_ or word.pos_ or '?'
    return text + '|' + tag


def represent_doc(doc):
    strings = []
    for sent in doc.sents:
        if sent.text.strip():
            words = ' '.join(represent_word(w) for w in sent if not w.is_space)
            strings.append(words)
    return '\n'.join(strings) + '\n' if strings else ''


@plac.annotations(
    in_file=("Path to input file", "positional", None, str),
    out_file=("Path to output file", "positional", None, str),
    spacy_model=("Name of spaCy model to use", "positional", None, str),
    n_workers=("Number of workers", "option", "n", int))
def main(in_file, out_file, spacy_model='en_core_web_sm', n_workers=4):
    input_path = Path(in_file)
    output_path = Path(out_file)
    if not input_path.exists():
        raise IOError("Can't find input file: {}".format(input_path))
    nlp = spacy.load(spacy_model)
    print("Using spaCy model {}".format(spacy_model))
    nlp.add_pipe(transform_doc, name='sense2vec')
    lines_count = 0
    with input_path.open('r', encoding='utf8') as texts:
        docs = nlp.pipe(texts, n_threads=n_workers)
        lines = (represent_doc(doc) for doc in docs)
        with output_path.open('w', encoding='utf8') as f:
            for line in tqdm(lines, desc='Lines', unit=''):
                lines_count += 1
                f.write(line)
    print("Successfully preprocessed {} lines".format(lines_count))
    print("{}".format(output_path.resolve()))


if __name__ == '__main__':
    plac.call(main)
