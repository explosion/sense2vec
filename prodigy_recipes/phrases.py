# coding: utf8
from __future__ import unicode_literals
import sys
from pathlib import Path

import prodigy
from prodigy.core import recipe_args
from prodigy.components.db import connect
from prodigy.components.sorters import Probability
from prodigy.util import log, prints, split_string, set_hashes
import requests
import sense2vec
from spacy.lang.en import English
import srsly


@prodigy.recipe('phrases.teach',
    dataset=recipe_args["dataset"],
    vectors_path=("Path to pretrained sense2vec vectors"),
    seeds=("One or more comma-separated seed terms", "option", "se", split_string),
    threshold=("Similarity threshold for sense2vec", "option", "t", float),
    batch_size=("Batch size for submitting annotations", "option", "bs", int),
    resume=("Resume from existing phrases dataset", "flag", "R", bool)
)
def phrases_teach(dataset, vectors_path, seeds, threshold=0.85, batch_size=5, resume=False):
    """
    Bootstrap a terminology list sense2vec. Prodigy
    will suggest similar terms based on the the most similar
    phrases from sense2vec
    """
    SENSES = ["auto", "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN",
            "NUM", "PART", "PERSON", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
            "VERB", "NORP", "FACILITY", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
            "WORK_OF_ART", "LANGUAGE"]
    
    print("Loading")
    LEMMATIZER = English().vocab.morphology.lemmatizer
    S2V = sense2vec.load(vectors_path)
    print("Loaded!")

    DB = connect()
    seed_tasks = [set_hashes({"text": s, "answer": "accept"}) for s in seeds]
    DB.add_examples(seed_tasks, datasets=[dataset])

    accept_phrases = seeds
    reject_phrases = []

    seen = set(accept_phrases)
    sensed = set()

    if resume:
        prev = DB.get_dataset(dataset)
        prev_accept = [eg["text"] for eg in prev if eg["answer"] == "accept"]
        prev_reject = [eg["text"] for eg in prev if eg["answer"] == "reject"]
        accept_phrases += prev_accept
        reject_phrases += prev_reject

        seen.update(set(accept_phrases))
        seen.update(set(reject_phrases))

    def format_for_s2v(word, sense):
        return word.replace(" ", "_") + "|" + sense

    def get_best(word, sense):
        if sense != "auto":  # if sense is specified, find respective entry
            if format_for_s2v(word, sense) in S2V:
                return (word, sense)
            return (None, None)
        freqs = []
        casings = [word, word.upper(), word.title()] if word.islower() else [word]
        for text in casings:  # try options
            for tag in SENSES:
                query = format_for_s2v(text, tag)
                if query in S2V:
                    freqs.append((S2V[query][0], (text, tag)))
        return max(freqs)[1] if freqs else (None, None)

    def get_similar(word, sense, n=100):
        query = format_for_s2v(word, sense)
        if query not in S2V:
            return []
        freq, query_vector = S2V[query]
        words, scores = S2V.most_similar(query_vector, n)
        words = [word.rsplit("|", 1) for word in words]
        # Don't know why we'd be getting unsensed entries, but fix.
        words = [entry for entry in words if len(entry) == 2]
        words = [(word.replace("_", " "), sense) for word, sense in words]
        return zip(words, scores)
    
    def find_similar(word: str, sense: str = "auto", n_results: int = 200):
        """Find similar terms for a given term and optional sense."""
        best_word, best_sense = get_best(word, sense)
        results = []
        if not word or not best_word:
            return results
        seen = set([best_word, min(LEMMATIZER(best_word, best_sense))])
        similar = get_similar(best_word, best_sense, n_results)
        for (word_entry, sense_entry), score in similar:
            head = min(LEMMATIZER(word_entry, sense_entry))
            if head not in seen and score > threshold:
                freq, _ = S2V[format_for_s2v(word_entry, sense_entry)]
                results.append((score, word_entry))
                seen.add(head)
            if len(results) >= n_results:
                break
        return results

    def update(answers):
        """Updates accept_phrases so that the stream can find new phrases"""
        for answer in answers:
            if answer['answer'] == 'accept':
                accept_phrases.append(answer['text'])
            elif answer['answer'] == 'reject':
                reject_phrases.append(answer['text'])
    
    def get_stream():
        """Continue querying sense2vec whenever we get a new phrase and presenting
        examples to the user with a similarity above the threshold parameter"""
        while True:
            seen.update(set([rp.lower() for rp in reject_phrases]))
            for p in accept_phrases:
                if p.lower() not in sensed:
                    sensed.add(p.lower())
                    for score, phrase in find_similar(p):
                        if phrase.lower() not in seen:
                            seen.add(phrase.lower())
                            yield {"text": phrase, 'meta': {'score': score}}

    stream = get_stream()

    return {
        'view_id': 'text',
        'dataset': dataset,
        'stream': stream,
        'update': update,
        'config': {
            'batch_size': batch_size
        }
    }


@prodigy.recipe(
    "phrases.to-patterns",
    dataset=recipe_args["dataset"],
    label=recipe_args["label"],
    output_file=recipe_args["output_file"],
)
def to_patterns(dataset=None, label=None, output_file=None):
    """
    Convert a list of seed phrases to a list of match patterns that can be used
    with ner.match. If no output file is specified, each pattern is printed
    so the recipe's output can be piped forward to ner.match.

    This is pretty much an exact copy of terms.to-patterns.
    The pattern for each example is just split on whitespace so instead of:

        {"label": "SHOE_BRAND", "pattern": [{"LOWER": "new balance"}]}


    which won't match anything you'll get:

        {"label": "SHOE_BRAND", "pattern": [{"LOWER": "new"}, {"LOWER": "balance"}]}
    """
    if label is None:
        prints(
            "--label is a required argument",
            "This is the label that will be assigned to all patterns "
            "created from terms collected in this dataset. ",
            exits=1,
            error=True,
        )

    DB = connect()

    def get_pattern(term, label):
        return {"label": label, "pattern": [{"lower": t.lower()} for t in term["text"].split()]}

    log("RECIPE: Starting recipe terms.to-patterns", locals())
    if dataset is None:
        log("RECIPE: Reading input terms from sys.stdin")
        terms = (srsly.json_loads(line) for line in sys.stdin)
    else:
        if dataset not in DB:
            prints("Can't find dataset '{}'".format(dataset), exits=1, error=True)
        terms = DB.get_dataset(dataset)
        log(
            "RECIPE: Reading {} input terms from dataset {}".format(len(terms), dataset)
        )
    if output_file:
        patterns = [
            get_pattern(term, label) for term in terms if term["answer"] == "accept"
        ]
        log("RECIPE: Generated {} patterns".format(len(patterns)))
        srsly.write_jsonl(output_file, patterns)
        prints("Exported {} patterns".format(len(patterns)), output_file)
    else:
        log("RECIPE: Outputting patterns")
        for term in terms:
            if term["answer"] == "accept":
                print(srsly.json_dumps(get_pattern(term, label)))
