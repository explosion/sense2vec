import prodigy
from prodigy.components.db import connect
from prodigy.util import log, split_string, set_hashes, TASK_HASH_ATTR
import murmurhash
from sense2vec import Sense2Vec
import srsly
import spacy
import random
from wasabi import Printer
from collections import defaultdict
import copy


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


@prodigy.recipe(
    "sense2vec.evaluate",
    dataset=("Dataset to save annotations to", "positional", None, str),
    vectors_path=("Path to pretrained sense2vec vectors", "positional", None, str),
    strategy=("Example selection strategy", "option", "st", str,),
    senses=("The senses to use (all if not set)", "option", "s", split_string),
    n_freq=("Number of most frequent entries to limit to", "option", "f", int),
    threshold=("Similarity threshold to consider examples", "option", "t", float),
    eval_whole=("Evaluate whole dataset instead of session", "flag", "E", bool),
    eval_only=("Don't annotate, only evaluate current set", "flag", "O", bool),
    show_scores=("Show all scores for debugging", "flag", "S", bool),
)
def evaluate(
    dataset,
    vectors_path,
    strategy="most_similar",
    senses=None,
    n_freq=100_000,
    threshold=0.7,
    eval_whole=False,
    eval_only=False,
    show_scores=False,
):
    """Evaluate a word vectors model by asking providing questions triples:
    is word A more similar to word B, or to word C? If the human mostly agrees
    with the model, the vectors model is good.
    """
    msg = Printer()
    random.seed(0)
    log("RECIPE: Starting recipe sense2vec.evaluate", locals())
    strategies = ["random", "most_similar"]
    if strategy not in strategies:
        msg.fail(f"Invalid strategy '{strategy}'. Expected: {strategies}", exits=1)
    s2v = Sense2Vec().from_disk(vectors_path)
    log("RECIPE: Loaded sense2vec vectors", vectors_path)

    def eval_dataset(set_id):
        """Output summary about user agreement with the model."""
        db = connect()
        data = db.get_dataset(set_id)
        accepted = [eg for eg in data if eg["answer"] == "accept" and eg.get("accept")]
        rejected = [eg for eg in data if eg["answer"] == "reject"]
        if not accepted and not rejected:
            msg.warn("No annotations collected", exits=1)
        high_conf = 0.8
        agree_count = 0
        disagree_high_conf = len([e for e in rejected if e["confidence"] > high_conf])
        for eg in accepted:
            choice = eg["accept"][0]
            score_choice = [o["score"] for o in eg["options"] if o["id"] == choice][0]
            score_other = [o["score"] for o in eg["options"] if o["id"] != choice][0]
            if score_choice > score_other:
                agree_count += 1
            elif eg["confidence"] > high_conf:
                disagree_high_conf += 1
        pc = agree_count / (len(accepted) + len(rejected))
        text = f"You agreed {agree_count} / {len(data)} times ({pc:.0%})"
        msg.info(f"Evaluating data from '{set_id}'")
        if pc > 0.5:
            msg.good(text)
        else:
            msg.fail(text)
        msg.text(f"You disagreed on {disagree_high_conf} high confidence scores")
        msg.text(f"You rejected {len(rejected)} suggestions as not similar")

    if eval_only:
        eval_dataset(dataset)
        return None

    def get_html(word, sense, score=None):
        html = f"{word} <strong style='opacity: 0.75; font-size: 14px; padding-left: 10px'>{sense}</strong>"
        if show_scores and score:
            html += f" <span style='opacity: 0.75; font-size: 12px; padding-left: 10px'>{score:.4}</span>"
        return html

    def get_stream():
        # Limit to most frequent entries
        keys = [key for key, _ in s2v.frequencies[:n_freq]]
        keys_by_sense = defaultdict(set)
        for key in keys:
            sense = s2v.split_key(key)[1]
            if senses is None or sense in senses:
                keys_by_sense[sense].add(key)
        keys_by_sense = {s: keys for s, keys in keys_by_sense.items() if len(keys) >= 3}
        all_senses = list(keys_by_sense.keys())
        total_keys = sum(len(keys) for keys in keys_by_sense.values())
        log(f"RECIPE: Using {total_keys} entries for {len(all_senses)} senses")
        while True:
            current_keys = copy.deepcopy(keys_by_sense)
            while any(len(values) >= 3 for values in current_keys.values()):
                sense = random.choice(all_senses)
                if strategy == "most_similar":
                    key_a = random.choice(list(current_keys[sense]))
                    most_similar = s2v.most_similar(key_a, n=100)
                    options = []
                    for key, score in most_similar:
                        if key in current_keys[sense]:
                            options.append((key, score))
                    if len(options) < 2:
                        continue
                    key_b, sim_ab = options[round(len(options) / 2)]
                    key_c, sim_ac = options[-1]
                else:
                    key_a, key_b, key_c = random.sample(current_keys[sense], 3)
                    sim_ab = s2v.similarity(key_a, key_b)
                    sim_ac = s2v.similarity(key_a, key_c)
                if len(set([key_a.lower(), key_b.lower(), key_c.lower()])) != 3:
                    continue
                if sim_ab < threshold or sim_ac < threshold:
                    continue
                current_keys[sense].remove(key_a)
                current_keys[sense].remove(key_b)
                current_keys[sense].remove(key_c)
                confidence = 1.0 - (min(sim_ab, sim_ac) / max(sim_ab, sim_ac))
                # Get a more representative hash
                task_hash = murmurhash.hash(" ".join([key_a] + sorted([key_b, key_c])))
                task = {
                    "label": "Which one is more similar?",
                    "html": get_html(*s2v.split_key(key_a)),
                    "key": key_a,
                    "options": [
                        {
                            "id": key_b,
                            "html": get_html(*s2v.split_key(key_b), sim_ab),
                            "score": sim_ab,
                        },
                        {
                            "id": key_c,
                            "html": get_html(*s2v.split_key(key_c), sim_ac),
                            "score": sim_ac,
                        },
                    ],
                    "confidence": confidence,
                    TASK_HASH_ATTR: task_hash,
                }
                if show_scores:
                    task["meta"] = {
                        "confidence": f"{confidence:.4}",
                        "strategy": strategy,
                    }
                yield task

    def on_exit(ctrl):
        set_id = dataset if eval_whole else ctrl.session_id
        eval_dataset(set_id)

    return {
        "view_id": "choice",
        "dataset": dataset,
        "stream": get_stream(),
        "on_exit": on_exit,
        "config": {"choice_style": "single", "choice_auto_accept": True},
    }
