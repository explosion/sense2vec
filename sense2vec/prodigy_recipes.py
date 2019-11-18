import prodigy
from prodigy.components.db import connect
from prodigy.util import log, split_string, set_hashes, TASK_HASH_ATTR, INPUT_HASH_ATTR
import murmurhash
from sense2vec import Sense2Vec
import srsly
import spacy
import random
from wasabi import msg
from collections import defaultdict, Counter
import copy
import catalogue


# fmt: off
eval_strategies = catalogue.create("prodigy", "sense2vec.eval")
EVAL_EXCLUDE_SENSES = ("SYM", "MONEY", "ORDINAL", "CARDINAL", "DATE", "TIME",
                       "PERCENT", "QUANTITY", "NUM", "X", "PUNCT")
# fmt: on


@prodigy.recipe(
    "sense2vec.teach",
    dataset=("Dataset to save annotations to", "positional", None, str),
    vectors_path=("Path to pretrained sense2vec vectors", "positional", None, str),
    seeds=("One or more comma-separated seed phrases", "option", "se", split_string),
    threshold=("Similarity threshold for sense2vec", "option", "t", float),
    n_similar=("Number of similar items to get at once", "option", "n", int),
    batch_size=("Batch size for submitting annotations", "option", "bs", int),
    case_sensitive=("Show the same terms with different casing", "flag", "CS", bool),
    resume=("Resume from existing phrases dataset", "flag", "R", bool),
)
def teach(
    dataset,
    vectors_path,
    seeds,
    threshold=0.85,
    n_similar=100,
    batch_size=5,
    case_sensitive=False,
    resume=False,
):
    """
    Bootstrap a terminology list using sense2vec. Prodigy will suggest similar
    terms based on the the most similar phrases from sense2vec, and the
    suggestions will be adjusted as you annotate and accept similar phrases. For
    each seed term, the best matching sense according to the sense2vec vectors
    will be used.

    If no similar terms are found above the given threshold, the threshold is
    lowered by 0.1 and similar terms are requested again.
    """
    log("RECIPE: Starting recipe sense2vec.teach", locals())
    s2v = Sense2Vec().from_disk(vectors_path)
    log("RECIPE: Loaded sense2vec vectors", vectors_path)
    html_template = "<span style='font-size: {{theme.largeText}}px'>{{word}}</span>"
    accept_keys = []
    seen = set()
    seed_tasks = []
    for seed in seeds:
        key = s2v.get_best_sense(seed)
        if key is None:
            msg.fail(f"Can't find seed term '{seed}' in vectors", exits=1)
        accept_keys.append(key)
        best_word, best_sense = s2v.split_key(key)
        seen.add(best_word if case_sensitive else best_word.lower())
        task = {
            "text": key,
            "word": best_word,
            "sense": best_sense,
            "meta": {"score": 1.0, "sense": best_sense},
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
        prev_accept_keys = [eg["text"] for eg in prev if eg["answer"] == "accept"]
        prev_words = [
            eg["word"] if case_sensitive else eg["word"].lower() for eg in prev
        ]
        accept_keys += prev_accept_keys
        seen.update(set(prev_words))
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
        nonlocal threshold
        while True:
            log(
                f"RECIPE: Looking for {n_similar} phrases most similar to "
                f"{len(accept_keys)} accepted keys"
            )
            most_similar = s2v.most_similar(accept_keys, n=n_similar)
            log(f"RECIPE: Found {len(most_similar)} most similar phrases")
            n_skipped = 0
            n_duplicate = 0
            for key, score in most_similar:
                if score > threshold:
                    word, sense = s2v.split_key(key)
                    if (case_sensitive and word in seen) or (
                        not case_sensitive and word.lower() in seen
                    ):
                        n_duplicate += 1
                        continue
                    seen.add(word if case_sensitive else word.lower())
                    # Make sure the score is a regular float, otherwise server
                    # may fail when trying to serialize it to/from JSON
                    meta = {"score": float(score), "sense": sense}
                    yield {"text": key, "word": word, "sense": sense, "meta": meta}
                else:
                    n_skipped += 1
            if n_skipped:
                log(f"RECIPE: Skipped {n_skipped} phrases below threshold {threshold}")
            if n_skipped == len(most_similar) - n_duplicate:
                # No most similar phrases were found that are above the
                # threshold, so lower the threshold if it's not already 0 or
                # return empty list so Prodigy shows "no tasks available"
                new_threshold = threshold - 0.1
                if new_threshold <= 0.0:
                    log(f"RECIPE: No suggestions for threshold {threshold:.2}")
                    return []
                log(
                    f"RECIPE: Lowering threshold from {threshold:.2} to {new_threshold:.2}"
                )
                threshold = new_threshold

    stream = get_stream()

    return {
        "view_id": "html",
        "dataset": dataset,
        "stream": stream,
        "update": update,
        "config": {"batch_size": batch_size, "html_template": html_template},
    }


@prodigy.recipe(
    "sense2vec.to-patterns",
    dataset=("Phrase dataset to convert", "positional", None, str),
    spacy_model=("spaCy model or blank:en (for tokenization)", "positional", None, str),
    label=("Label to apply to all patterns", "positional", None, str),
    output_file=("Optional output file. Defaults to stdout", "option", "o", str),
    case_sensitive=("Make patterns case-sensitive", "flag", "CS", bool),
    dry=("Perform a dry run and don't output anything", "flag", "D", bool),
)
def to_patterns(
    dataset, spacy_model, label, output_file="-", case_sensitive=False, dry=False
):
    """
    Convert a dataset of phrases collected with sense2vec.teach to token-based
    match patterns that can be used with spaCy's EntityRuler or recipes like
    ner.match. If no output file is specified, the patterns are written to
    stdout. The examples are tokenized so that multi-token terms are represented
    correctly, e.g.:
    {"label": "SHOE_BRAND", "pattern": [{"LOWER": "new"}, {"LOWER": "balance"}]}

    For tokenization, you can either pass in the name of a spaCy model (e.g. if
    you're using a model with custom tokenization), or "blank:" plus the
    language code you want to use, e.g. blank:en or blank:de. Make sure to use
    the same language / tokenizer you're planning to use at runtime – otherwise
    your patterns may not match.
    """
    log("RECIPE: Starting recipe sense2vec.to-patterns", locals())
    if spacy_model.startswith("blank:"):
        nlp = spacy.blank(spacy_model.replace("blank:", ""))
    else:
        nlp = spacy.load(spacy_model)
    log(f"RECIPE: Loaded spaCy model '{spacy_model}'")
    DB = connect()
    if dataset not in DB:
        msg.fail(f"Can't find dataset '{dataset}'", exits=1)
    examples = DB.get_dataset(dataset)
    terms = set([eg["word"] for eg in examples if eg["answer"] == "accept"])
    if case_sensitive:
        patterns = [[{"text": t.lower_} for t in nlp.make_doc(term)] for term in terms]
    else:
        terms = set([word.lower() for word in terms])
        patterns = [[{"lower": t.lower_} for t in nlp.make_doc(term)] for term in terms]
    patterns = [{"label": label, "pattern": pattern} for pattern in patterns]
    log(f"RECIPE: Generated {len(patterns)} patterns")
    if not dry:
        srsly.write_jsonl(output_file, patterns)
    return patterns


@prodigy.recipe(
    "sense2vec.eval",
    dataset=("Dataset to save annotations to", "positional", None, str),
    vectors_path=("Path to pretrained sense2vec vectors", "positional", None, str),
    strategy=("Example selection strategy", "option", "st", str,),
    senses=("The senses to use (all if not set)", "option", "s", split_string),
    exclude_senses=("The senses to exclude", "option", "es", split_string),
    n_freq=("Number of most frequent entries to limit to", "option", "f", int),
    threshold=("Similarity threshold to consider examples", "option", "t", float),
    batch_size=("The batch size to use", "option", "b", int),
    eval_whole=("Evaluate whole dataset instead of session", "flag", "E", bool),
    eval_only=("Don't annotate, only evaluate current set", "flag", "O", bool),
    show_scores=("Show all scores for debugging", "flag", "S", bool),
)
def evaluate(
    dataset,
    vectors_path,
    strategy="most_similar",
    senses=None,
    exclude_senses=EVAL_EXCLUDE_SENSES,
    n_freq=100_000,
    threshold=0.7,
    batch_size=10,
    eval_whole=False,
    eval_only=False,
    show_scores=False,
):
    """
    Evaluate a sense2vec model by asking about phrase triples: is word A more
    similar to word B, or to word C? If the human mostly agrees with the model,
    the vectors model is good.
    """
    random.seed(0)
    log("RECIPE: Starting recipe sense2vec.eval", locals())
    strategies = eval_strategies.get_all()
    if strategy not in strategies.keys():
        err = f"Invalid strategy '{strategy}'. Expected: {list(strategies.keys())}"
        msg.fail(err, exits=1)
    s2v = Sense2Vec().from_disk(vectors_path)
    log("RECIPE: Loaded sense2vec vectors", vectors_path)

    def get_html(key, score=None, large=False):
        word, sense = s2v.split_key(key)
        html_word = f"<span style='font-size: {30 if large else 20}px'>{word}</span>"
        html_sense = f"<strong style='opacity: 0.75; font-size: 14px; padding-left: 10px'>{sense}</strong>"
        html = f"{html_word} {html_sense}"
        if show_scores and score is not None:
            html += f" <span style='opacity: 0.75; font-size: 12px; padding-left: 10px'>{score:.4}</span>"
        return html

    def get_stream():
        strategy_func = eval_strategies.get(strategy)
        log(f"RECIPE: Using strategy {strategy}")
        # Limit to most frequent entries
        keys = [key for key, _ in s2v.frequencies[:n_freq]]
        keys_by_sense = defaultdict(set)
        for key in keys:
            try:
                sense = s2v.split_key(key)[1]
            except ValueError:
                continue
            if (senses is None or sense in senses) and sense not in exclude_senses:
                keys_by_sense[sense].add(key)
        keys_by_sense = {s: keys for s, keys in keys_by_sense.items() if len(keys) >= 3}
        all_senses = list(keys_by_sense.keys())
        total_keys = sum(len(keys) for keys in keys_by_sense.values())
        log(f"RECIPE: Using {total_keys} entries for {len(all_senses)} senses")
        n_passes = 1
        while True:
            log(f"RECIPE: Iterating over the data ({n_passes})")
            current_keys = copy.deepcopy(keys_by_sense)
            while any(len(values) >= 3 for values in current_keys.values()):
                sense = random.choice(all_senses)
                all_keys = list(current_keys[sense])
                key_a, key_b, key_c, sim_ab, sim_ac = strategy_func(s2v, all_keys)
                if len(set([key_a.lower(), key_b.lower(), key_c.lower()])) != 3:
                    continue
                if sim_ab < threshold or sim_ac < threshold:
                    continue
                for key in (key_a, key_b, key_c):
                    current_keys[sense].remove(key)
                confidence = 1.0 - (min(sim_ab, sim_ac) / max(sim_ab, sim_ac))
                input_hash = murmurhash.hash(key_a)
                task_hash = murmurhash.hash(" ".join([key_a] + sorted([key_b, key_c])))
                task = {
                    "label": "Which one is more similar?",
                    "html": get_html(key_a, large=True),
                    "text": f"{key_a}: {key_b}, {key_c}",
                    "key": key_a,
                    "options": [
                        {
                            "id": key_b,
                            "html": get_html(key_b, sim_ab),
                            "score": sim_ab,
                        },
                        {
                            "id": key_c,
                            "html": get_html(key_c, sim_ac),
                            "score": sim_ac,
                        },
                    ],
                    "confidence": confidence,
                    TASK_HASH_ATTR: task_hash,
                    INPUT_HASH_ATTR: input_hash,
                }
                if show_scores:
                    task["meta"] = {
                        "confidence": f"{confidence:.4}",
                        "strategy": strategy,
                    }
                yield task
            n_passes += 1

    def eval_dataset(set_id):
        """Output summary about user agreement with the model."""
        DB = connect()
        data = DB.get_dataset(set_id)
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

    def on_exit(ctrl):
        set_id = dataset if eval_whole else ctrl.session_id
        eval_dataset(set_id)

    if eval_only:
        eval_dataset(dataset)
        return None

    return {
        "view_id": "choice",
        "dataset": dataset,
        "stream": get_stream(),
        "on_exit": on_exit,
        "config": {
            "batch_size": batch_size,
            "choice_style": "single",
            "choice_auto_accept": True,
        },
    }


@eval_strategies.register("random")
def eval_strategy_random(s2v, keys):
    key_a, key_b, key_c = random.sample(keys, 3)
    sim_ab = s2v.similarity(key_a, key_b)
    sim_ac = s2v.similarity(key_a, key_c)
    return key_a, key_b, key_c, sim_ab, sim_ac


@eval_strategies.register("most_similar")
def eval_strategy_most_similar(s2v, keys):
    key_a = random.choice(keys)
    most_similar = s2v.most_similar(key_a, n=min(2000, len(s2v)))
    options = [(key, score) for key, score in most_similar if key in keys]
    if len(options) < 2:
        return eval_strategy_most_similar(s2v, keys)
    key_b, sim_ab = options[len(options) // 2]
    key_c, sim_ac = options[-1]
    return key_a, key_b, key_c, sim_ab, sim_ac


@eval_strategies.register("most_least_similar")
def eval_strategy_most_least_similar(s2v, keys):
    n_similar = 100
    key_a = random.choice(keys)
    most_similar_a = s2v.most_similar(key_a, n=n_similar)
    options_a = [(key, score) for key, score in most_similar_a if key in keys]
    if len(options_a) < 1:
        return eval_strategy_most_least_similar(s2v, keys)
    key_b, sim_ab = options_a[-1]
    most_similar_b = s2v.most_similar(key_b, n=n_similar)
    options_b = [(key, score) for key, score in most_similar_b if key in keys]
    if len(options_b) < 1:
        return eval_strategy_most_least_similar(s2v, keys)
    key_c, sim_ac = options_b[-1]
    return key_a, key_b, key_c, sim_ab, sim_ac


@prodigy.recipe(
    "sense2vec.eval-most-similar",
    dataset=("Dataset to save annotations to", "positional", None, str),
    vectors_path=("Path to pretrained sense2vec vectors", "positional", None, str),
    senses=("The senses to use (all if not set)", "option", "s", split_string),
    exclude_senses=("The senses to exclude", "option", "es", split_string),
    n_freq=("Number of most frequent entries to limit to", "option", "f", int),
    n_similar=("Number of similar items to check", "option", "n", int),
    batch_size=("The batch size to use", "option", "b", int),
    eval_whole=("Evaluate whole dataset instead of session", "flag", "E", bool),
    eval_only=("Don't annotate, only evaluate current set", "flag", "O", bool),
    show_scores=("Show all scores for debugging", "flag", "S", bool),
)
def eval_most_similar(
    dataset,
    vectors_path,
    senses=None,
    exclude_senses=EVAL_EXCLUDE_SENSES,
    n_freq=100_000,
    n_similar=10,
    batch_size=5,
    eval_whole=False,
    eval_only=False,
    show_scores=False,
):
    """
    Evaluate a vectors model by looking at the most similar entries it returns
    for a random phrase and unselecting the mistakes.
    """
    log("RECIPE: Starting recipe sense2vec.eval-most-similar", locals())
    random.seed(0)
    s2v = Sense2Vec().from_disk(vectors_path)
    log("RECIPE: Loaded sense2vec vectors", vectors_path)
    seen = set()
    DB = connect()
    if dataset in DB:
        examples = DB.get_dataset(dataset)
        seen.update([eg["text"] for eg in examples if eg["answer"] == "accept"])
        log(f"RECIPE: Skipping {len(seen)} terms already in dataset")

    def get_html(key, score=None, large=False):
        word, sense = s2v.split_key(key)
        html_word = f"<span style='font-size: {30 if large else 20}px'>{word}</span>"
        html_sense = f"<strong style='opacity: 0.75; font-size: 14px; padding-left: 10px'>{sense}</strong>"
        html = f"{html_word} {html_sense}"
        if show_scores and score is not None:
            html += f" <span style='opacity: 0.75; font-size: 12px; padding-left: 10px'>{score:.4}</span>"
        return html

    def get_stream():
        keys = [key for key, _ in s2v.frequencies[:n_freq] if key not in seen]
        while len(keys):
            key = random.choice(keys)
            keys.remove(key)
            word, sense = s2v.split_key(key)
            if sense in exclude_senses or (senses is not None and sense not in senses):
                continue
            most_similar = s2v.most_similar(key, n=n_similar)
            options = [{"id": k, "html": get_html(k, s)} for k, s in most_similar]
            task_hash = murmurhash.hash(key)
            task = {
                "html": get_html(key, large=True),
                "text": key,
                "options": options,
                "accept": [key for key, _ in most_similar],  # pre-select all
                TASK_HASH_ATTR: task_hash,
                INPUT_HASH_ATTR: task_hash,
            }
            yield task

    def eval_dataset(set_id):
        DB = connect()
        data = DB.get_dataset(set_id)
        accepted = [eg for eg in data if eg["answer"] == "accept" and eg.get("accept")]
        rejected = [eg for eg in data if eg["answer"] == "reject"]
        ignored = [eg for eg in data if eg["answer"] == "ignore"]
        if not accepted and not rejected:
            msg.warn("No annotations collected", exits=1)
        total_count = 0
        agree_count = 0
        for eg in accepted:
            total_count += len(eg.get("options", []))
            agree_count += len(eg.get("accept", []))
        msg.info(f"Evaluating data from '{set_id}'")
        msg.text(f"You rejected {len(rejected)} and ignored {len(ignored)} pair(s)")
        pc = agree_count / total_count
        text = f"You agreed {agree_count} / {total_count} times ({pc:.0%})"
        if pc > 0.5:
            msg.good(text)
        else:
            msg.fail(text)

    def on_exit(ctrl):
        set_id = dataset if eval_whole else ctrl.session_id
        eval_dataset(set_id)

    if eval_only:
        eval_dataset(dataset)
        return None

    return {
        "view_id": "choice",
        "dataset": dataset,
        "stream": get_stream(),
        "on_exit": on_exit,
        "config": {"choice_style": "multiple", "batch_size": batch_size},
    }


@prodigy.recipe(
    "sense2vec.eval-ab",
    dataset=("Dataset to save annotations to", "positional", None, str),
    vectors_path_a=("Path to pretrained sense2vec vectors", "positional", None, str),
    vectors_path_b=("Path to pretrained sense2vec vectors", "positional", None, str),
    senses=("The senses to use (all if not set)", "option", "s", split_string),
    exclude_senses=("The senses to exclude", "option", "es", split_string),
    n_freq=("Number of most frequent entries to limit to", "option", "f", int),
    batch_size=("The batch size to use", "option", "b", int),
    eval_whole=("Evaluate whole dataset instead of session", "flag", "E", bool),
    eval_only=("Don't annotate, only evaluate current set", "flag", "O", bool),
    show_mapping=("Show A/B mapping for debugging", "flag", "S", bool),
)
def eval_ab(
    dataset,
    vectors_path_a,
    vectors_path_b,
    senses=None,
    exclude_senses=EVAL_EXCLUDE_SENSES,
    n_freq=100_000,
    n_similar=10,
    batch_size=5,
    eval_whole=False,
    eval_only=False,
    show_mapping=False,
):
    """
    Perform an A/B evaluation of two pretrained sense2vec vector models by
    comparing the most similar entries they return for a random phrase. The
    UI shows two randomized options with the most similar entries of each model
    and highlights the phrases that differ. At the end of the annotation
    session the overall stats and preferred model are shown.
    """
    log("RECIPE: Starting recipe sense2vec.eval-ab", locals())
    random.seed(0)
    s2v_a = Sense2Vec().from_disk(vectors_path_a)
    s2v_b = Sense2Vec().from_disk(vectors_path_b)
    mapping = {"A": vectors_path_a, "B": vectors_path_b}
    log("RECIPE: Loaded sense2vec vectors", (vectors_path_a, vectors_path_b))
    seen = set()
    DB = connect()
    if dataset in DB:
        examples = DB.get_dataset(dataset)
        seen.update([eg["text"] for eg in examples if eg["answer"] == "accept"])
        log(f"RECIPE: Skipping {len(seen)} terms already in dataset")

    def get_term_html(key):
        word, sense = s2v_a.split_key(key)
        return (
            f"<span style='font-size: 30px'>{word}</span> "
            f"<strong style='opacity: 0.75; font-size: 14px; padding-left: "
            f"10px'>{sense}</strong>"
        )

    def get_option_html(most_similar, overlap):
        html = []
        for key in most_similar:
            font_weight = "normal" if key in overlap else "bold"
            border_color = "#f6f6f6" if key in overlap else "#ccc"
            word, sense = s2v_a.split_key(key)
            html.append(
                f"<span style='display: inline-block; background: #f6f6f6; "
                f"font-weight: {font_weight}; border: 1px solid {border_color}; "
                f"padding: 0 8px; margin: 0 5px 5px 0; border-radius: 5px; "
                f"white-space: nowrap'>{word} <span style='font-weight: bold; "
                f"text-transform: uppercase; font-size: 10px; margin-left: "
                f"5px'>{sense}</span></span>"
            )
        html = " ".join(html) if html else "<em>No results</em>"
        return (
            f"<div style='font-size: 16px; line-height: 1.75; padding: 5px "
            f"12px 5px 0'>{html}</div>"
        )

    def get_stream():
        keys_a = [key for key, _ in s2v_a.frequencies[:n_freq] if key not in seen]
        keys_b = [key for key, _ in s2v_b.frequencies[:n_freq] if key not in seen]
        while len(keys_a):
            key = random.choice(keys_a)
            keys_a.remove(key)
            word, sense = s2v_a.split_key(key)
            if sense in exclude_senses or (senses is not None and sense not in senses):
                continue
            if key not in keys_b:
                continue
            similar_a = set([k for k, _ in s2v_a.most_similar(key, n=n_similar)])
            similar_b = set([k for k, _ in s2v_b.most_similar(key, n=n_similar)])
            overlap = similar_a.intersection(similar_b)
            options = [
                {"id": "A", "html": get_option_html(similar_a, overlap)},
                {"id": "B", "html": get_option_html(similar_b, overlap)},
            ]
            random.shuffle(options)
            task_hash = murmurhash.hash(key)
            task = {
                "html": get_term_html(key),
                "text": key,
                "options": options,
                TASK_HASH_ATTR: task_hash,
                INPUT_HASH_ATTR: task_hash,
            }
            if show_mapping:
                opt_map = [f"{opt['id']} ({mapping[opt['id']]})" for opt in options]
                task["meta"] = {i + 1: opt for i, opt in enumerate(opt_map)}
            yield task

    def eval_dataset(set_id):
        DB = connect()
        data = DB.get_dataset(set_id)
        accepted = [eg for eg in data if eg["answer"] == "accept" and eg.get("accept")]
        rejected = [eg for eg in data if eg["answer"] == "reject"]
        ignored = [eg for eg in data if eg["answer"] == "ignore"]
        if not accepted and not rejected:
            msg.warn("No annotations collected", exits=1)
        counts = Counter()
        for eg in accepted:
            for model_id in eg["accept"]:
                counts[model_id] += 1
        preference, _ = counts.most_common(1)[0]
        ratio = f"{counts[preference]} / {sum(counts.values()) - counts[preference]}"
        msg.info(f"Evaluating data from '{set_id}'")
        msg.text(f"You rejected {len(rejected)} and ignored {len(ignored)} pair(s)")
        if counts["A"] == counts["B"]:
            msg.warn(f"No preference ({ratio})")
        else:
            pc = counts[preference] / sum(counts.values())
            msg.good(f"You preferred vectors {preference} with {ratio} ({pc:.0%})")
            msg.text(mapping[preference])

    def on_exit(ctrl):
        set_id = dataset if eval_whole else ctrl.session_id
        eval_dataset(set_id)

    if eval_only:
        eval_dataset(dataset)
        return None

    return {
        "view_id": "choice",
        "dataset": dataset,
        "stream": get_stream(),
        "on_exit": on_exit,
        "config": {
            "batch_size": batch_size,
            "choice_style": "single",
            "choice_auto_accept": True,
        },
    }
