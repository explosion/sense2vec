from __future__ import print_function, unicode_literals, division
import io
import bz2
import logging
from toolz import partition
from os import path
import os
import re

import spacy.en
from preshed.counter import PreshCounter
from spacy.tokens.doc import Doc

from joblib import Parallel, delayed
import plac
try:
    import ujson as json
except ImportError:
    import json


LABELS = {
    'ENT': 'ENT',
    'PERSON': 'ENT',
    'NORP': 'ENT',
    'FAC': 'ENT',
    'ORG': 'ENT',
    'GPE': 'ENT',
    'LOC': 'ENT',
    'LAW': 'ENT',
    'PRODUCT': 'ENT',
    'EVENT': 'ENT',
    'WORK_OF_ART': 'ENT',
    'LANGUAGE': 'ENT',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': 'QUANTITY',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'CARDINAL'
}


def parallelize(func, iterator, n_jobs, extra):
    extra = tuple(extra)
    return Parallel(n_jobs=n_jobs)(delayed(func)(*(item + extra)) for item in iterator)


def iter_comments(loc):
    with bz2.BZ2File(loc) as file_:
        for i, line in enumerate(file_):
            yield ujson.loads(line)['body']


pre_format_re = re.compile(r'^[\`\*\~]')
post_format_re = re.compile(r'[\`\*\~]$')
url_re = re.compile(r'\[([^]]+)\]\(%%URL\)')
link_re = re.compile(r'\[([^]]+)\]\(https?://[^\)]+\)')
def strip_meta(text):
    text = link_re.sub(r'\1', text)
    text = text.replace('&gt;', '>').replace('&lt;', '<')
    text = pre_format_re.sub('', text)
    text = post_format_re.sub('', text)
    return text


def load_and_transform(batch_id, in_loc, out_dir):
    out_loc = path.join(out_dir, '%d.txt' % batch_id)
    if path.exists(out_loc):
        return None
    print('Batch', batch_id)
    nlp = spacy.en.English(parser=False, tagger=False, matcher=False, entity=False)
    with io.open(out_loc, 'w', encoding='utf8') as out_file:
        with io.open(in_loc, 'rb') as in_file:
            for byte_string in Doc.read_bytes(in_file):
                doc = Doc(nlp.vocab).from_bytes(byte_string)
                doc.is_parsed = True
                out_file.write(transform_doc(doc)) 


def parse_and_transform(batch_id, input_, out_dir):
    out_loc = path.join(out_dir, '%d.txt' % batch_id)
    if path.exists(out_loc):
        return None
    print('Batch', batch_id)
    nlp = spacy.en.English()
    nlp.matcher = None
    with io.open(out_loc, 'w', encoding='utf8') as file_:
        for text in input_:
            file_.write(transform_doc(nlp(strip_meta(text))))


def transform_doc(doc):
    for ent in doc.ents:
        ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
            np = np[1:]
        np.merge(np.root.tag_, np.text, np.root.ent_type_)
    strings = []
    for sent in doc.sents:
        if sent.text.strip():
            strings.append(' '.join(represent_word(w) for w in sent if not w.is_space))
    if strings:
        return '\n'.join(strings) + '\n'
    else:
        return ''


def represent_word(word):
    if word.like_url:
        return '%%URL|X'
    text = re.sub(r'\s', '_', word.text)
    tag = LABELS.get(word.ent_type_, word.pos_)
    if not tag:
        tag = '?'
    return text + '|' + tag


@plac.annotations(
    in_loc=("Location of input file"),
    out_dir=("Location of input file"),
    n_workers=("Number of workers", "option", "n", int),
    load_parses=("Load parses from binary", "flag", "b"),
)
def main(in_loc, out_dir, n_workers=4, load_parses=False):
    if not path.exists(out_dir):
        path.join(out_dir)
    if load_parses:
        jobs = [path.join(in_loc, fn) for fn in os.listdir(in_loc)]
        do_work = load_and_transform
    else:
        jobs = partition(200000, iter_comments(in_loc))
        do_work = parse_and_transform
    parallelize(do_work, enumerate(jobs), n_workers, [out_dir])
 

if __name__ == '__main__':
    plac.call(main)
