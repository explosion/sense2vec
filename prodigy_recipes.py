import prodigy
from prodigy.components.db import connect
from prodigy.util import log, split_string, set_hashes
from sense2vec import Sense2Vec
import srsly
import spacy


@prodigy.recipe(
    "sense2vec.teach",
    dataset=("Dataset to save annotations to", "positional", None, str),
    vectors_path=("Path to pretrained sense2vec vectors", "positional", None, str),
    seeds=("One or more comma-separated seed terms", "option", "se", split_string),
    threshold=("Similarity threshold for sense2vec", "option", "t", float),
    top_n=("Only get the top n results for each accepted term", "option", "n", int),
    batch_size=("Batch size for submitting annotations", "option", "bs", int),
    resume=("Resume from existing phrases dataset", "flag", "R", bool),
)
def teach(
    dataset, vectors_path, seeds, threshold=0.85, top_n=200, batch_size=5, resume=False
):
    """
    Bootstrap a terminology list sense2vec. Prodigy will suggest similar terms
    based on the the most similar phrases from sense2vec.
    """
    log("RECIPE: Starting recipe sense2vec.teach", locals())
    s2v = Sense2Vec().from_disk(vectors_path)
    log("RECIPE: Loaded sense2vec", locals())
    seed_keys = []
    for seed in seeds:
        best_word, best_sense = s2v.get_best_sense(seed)
        if best_sense is None:
            raise ValueError(f"Can't find seed term '{seed}' in vectors")
        seed_keys.append(s2v.make_key(best_word, best_sense))
    print(f"Starting with seed keys: {seed_keys}")
    DB = connect()
    seed_tasks = [set_hashes({"text": s, "answer": "accept"}) for s in seed_keys]
    DB.add_examples(seed_tasks, datasets=[dataset])
    accept_keys = seed_keys
    reject_keys = []
    seen = set(accept_keys)

    if resume:
        prev = DB.get_dataset(dataset)
        prev_accept = [eg["text"] for eg in prev if eg["answer"] == "accept"]
        prev_reject = [eg["text"] for eg in prev if eg["answer"] == "reject"]
        accept_keys += prev_accept
        reject_keys += prev_reject
        seen.update(set(accept_keys))
        seen.update(set(reject_keys))
        log(f"RECIPE: Resuming from {len(prev)} previous examples in dataset {dataset}")

    def update(answers):
        """Updates accept_keys so that the stream can find new phrases."""
        log(f"RECIPE: Updating with {len(answers)} answers")
        for answer in answers:
            phrase = answer["text"]
            if answer["answer"] == "accept":
                accept_keys.append(phrase)
            elif answer["answer"] == "reject":
                reject_keys.append(phrase)

    def get_stream():
        """Continue querying sense2vec whenever we get a new phrase and
        presenting examples to the user with a similarity above the threshold
        parameter."""
        while True:
            log(f"RECIPE: Getting {top_n} similar phrases")
            most_similar = s2v.most_similar(accept_keys, n_similar=top_n)
            for key, score in most_similar:
                if key not in seen and score > threshold:
                    seen.add(key)
                    word, sense = s2v.split_key(key)
                    yield {
                        "text": key,
                        "word": word,
                        "sense": sense,
                        "meta": {"score": score},
                    }

    stream = get_stream()

    return {
        "view_id": "html",
        "dataset": dataset,
        "stream": stream,
        "update": update,
        "config": {"batch_size": batch_size, "html_template": "{{word}} ({{sense}})"},
    }


@prodigy.recipe(
    "sense2vec.to-patterns",
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("spaCy model for tokenization", "positional", None, str),
    label=("Label to apply to all patterns", "positional", None, str),
    output_file=("Optional output file. Defaults to stdout", "option", "o", str),
)
def to_patterns(dataset, spacy_model, label, output_file="-"):
    """
    Convert a list of seed phrases to a list of match patterns that can be used
    with ner.match. If no output file is specified, each pattern is printed.
    The examples are tokenized to make sure that multi-token terms are
    represented correctly, e.g.:
    {"label": "SHOE_BRAND", "pattern": [{"LOWER": "new"}, {"LOWER": "balance"}]}
    """
    log("RECIPE: Starting recipe sense2vec.to-patterns", locals())
    nlp = spacy.load(spacy_model)
    log(f"RECIPE: Loaded spaCy model '{spacy_model}'")
    DB = connect()
    examples = DB.get_dataset(dataset)
    terms = [eg["text"] for eg in examples if eg["answer"] == "accept"]
    patterns = [{"lower": t.lower_ for t in nlp.make_doc(term)} for term in terms]
    patterns = [{"label": label, "pattern": pattern} for pattern in patterns]
    log(f"RECIPE: Generated {len(patterns)} patterns")
    srsly.write_jsonl(output_file, patterns)
