# coding: utf8
from __future__ import unicode_literals


def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result


def transform_doc(doc):
    """
    Transform a spaCy Doc to match the sense2vec format: merge entities
    into one token and merge noun chunks without determiners.
    """
    spans = list(doc.ents)
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ("advmod", "amod", "compound"):
            np = np[1:]
        spans.append(np)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            root = span.root
            attrs = {"tag": root.tag_, "lemma": root.lemma_, "ent_type": root.ent_type_}
            retokenizer.merge(span, attrs=attrs)
    return doc
