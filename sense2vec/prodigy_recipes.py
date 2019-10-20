import prodigy
from prodigy.components.db import connect
from prodigy.util import log, split_string, set_hashes, TASK_HASH_ATTR
from sense2vec import Sense2Vec
import srsly
import spacy


HTML_TEMPLATE = """
<span style="font-size: {{theme.largeText}}px">{{word}}</span>
<strong style="opacity: 0.75">{{sense}}</strong>
"""


@prodigy.recipe(
    "sense2vec.teach",
    dataset=("Dataset to save annotations to", "positional", None, str),
    vectors_path=("Path to pretrained sense2vec vectors", "positional", None, str),
    seeds=("One or more comma-separated seed phrases", "option", "se", split_string),
    threshold=("Similarity threshold for sense2vec", "option", "t", float),
    n_similar=("Number of similar items to get at once", "option", "n", int),
    batch_size=("Batch size for submitting annotations", "option", "bs", int),
    resume=("Resume from existing phrases dataset", "flag", "R", bool),
)
def teach(
    dataset,
    vectors_path,
    seeds,
    threshold=0.85,
    n_similar=20,
    batch_size=5,
    resume=False,
):
    """
    Bootstrap a terminology list using sense2vec. Prodigy will suggest similar
    terms based on the the most similar phrases from sense2vec, and the
    suggestions will be adjusted as you annotate and accept similar phrases. For
    each seed term, the best matching sense according to the sense2vec vectors
    will be used.
    """
    log("RECIPE: Starting recipe sense2vec.teach", locals())
    s2v = Sense2Vec().from_disk(vectors_path)
    log("RECIPE: Loaded sense2vec vectors", vectors_path)
    accept_keys = []
    seen = set(accept_keys)
    seed_tasks = []
    for seed in seeds:
        key = s2v.get_best_sense(seed)
        if key is None:
            raise ValueError(f"Can't find seed term '{seed}' in vectors")
        accept_keys.append(key)
        best_word, best_sense = s2v.split_key(key)
        task = {
            "text": key,
            "word": best_word,
            "sense": best_sense,
            "meta": {"score": 1.0},
            "answer": "accept",
        }
        seed_tasks.append(set_hashes(task))
    print(f"Starting with seed keys: {accept_keys}")
    DB = connect()
    if dataset not in DB:
        DB.add_dataset(dataset)
    dataset_hashes = DB.get_task_hashes(dataset)
    DB.add_examples(
        [st for st in seed_tasks if st[TASK_HASH_ATTR] not in dataset_hashes],
        datasets=[dataset],
    )

    if resume:
        prev = DB.get_dataset(dataset)
        prev_accept = [eg["text"] for eg in prev if eg["answer"] == "accept"]
        accept_keys += prev_accept
        seen.update(set(accept_keys))
        log(f"RECIPE: Resuming from {len(prev)} previous examples in dataset {dataset}")

    def update(answers):
        """Updates accept_keys so that the stream can find new phrases."""
        log(f"RECIPE: Updating with {len(answers)} answers")
        for answer in answers:
            phrase = answer["text"]
            if answer["answer"] == "accept":
                accept_keys.append(phrase)

    def get_stream():
        """Continue querying sense2vec whenever we get a new phrase and
        presenting examples to the user with a similarity above the threshold
        parameter."""
        while True:
            log(
                f"RECIPE: Looking for {n_similar} phrases most similar to "
                f"{len(accept_keys)} accepted keys"
            )
            most_similar = s2v.most_similar(accept_keys, n=n_similar)
            log(f"RECIPE: Found {len(most_similar)} most similar phrases")
            for key, score in most_similar:
                if key not in seen and score > threshold:
                    seen.add(key)
                    word, sense = s2v.split_key(key)
                    # Make sure the score is a regular float, otherwise server
                    # may fail when trying to serialize it to/from JSON
                    meta = {"score": float(score)}
                    yield {"text": key, "word": word, "sense": sense, "meta": meta}

    stream = get_stream()

    return {
        "view_id": "html",
        "dataset": dataset,
        "stream": stream,
        "update": update,
        "config": {"batch_size": batch_size, "html_template": HTML_TEMPLATE},
    }


@prodigy.recipe(
    "sense2vec.to-patterns",
    dataset=("Phrase dataset to convert", "positional", None, str),
    spacy_model=("spaCy model for tokenization", "positional", None, str),
    label=("Label to apply to all patterns", "positional", None, str),
    output_file=("Optional output file. Defaults to stdout", "option", "o", str),
    case_sensitive=("Make patterns case-sensitive", "flag", "CS", bool),
    dry=("Perform a dry run and don't output anything", "flag", "D", bool),
)
def to_patterns(
    dataset, spacy_model, label, output_file="-", case_sensitive=False, dry=False
):
    """
    Convert a list of seed phrases to a list of token-based match patterns that
    can be used with spaCy's EntityRuler or recipes like ner.match. If no output
    file is specified, the patterns are written to stdout. The examples are
    tokenized so that multi-token terms are represented correctly, e.g.:
    {"label": "SHOE_BRAND", "pattern": [{"LOWER": "new"}, {"LOWER": "balance"}]}
    """
    log("RECIPE: Starting recipe sense2vec.to-patterns", locals())
    nlp = spacy.load(spacy_model)
    log(f"RECIPE: Loaded spaCy model '{spacy_model}'")
    DB = connect()
    if dataset not in DB:
        raise ValueError(f"Can't find dataset '{dataset}'")
    examples = DB.get_dataset(dataset)
    terms = [eg["text"] for eg in examples if eg["answer"] == "accept"]
    if case_sensitive:
        patterns = [{"text": t.text for t in nlp.make_doc(term)} for term in terms]
    else:
        patterns = [{"lower": t.lower_ for t in nlp.make_doc(term)} for term in terms]
    patterns = [{"label": label, "pattern": pattern} for pattern in patterns]
    log(f"RECIPE: Generated {len(patterns)} patterns")
    if not dry:
        srsly.write_jsonl(output_file, patterns)
    return patterns
