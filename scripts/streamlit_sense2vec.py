"""
Streamlit script for visualizing most similar sense2vec entries

Lets you look up words and an optional sense (sense with the highest frequency
is used if "auto" is selected) and shows the N most similar phrases, their
scores and their frequencies.

To add vector models, you can pass one or more directory paths (containing the
serialized sense2vec components) when you run it with "streamlit run":
streamlit run streamlit_sense2vec.py /path/to/sense2vec /path/to/other_sense2vec
"""
import streamlit as st
from sense2vec import Sense2Vec
import sys

SENSE2VEC_PATHS = list(sys.argv[1:])
DEFAULT_WORD = "natural language processing"


@st.cache(allow_output_mutation=True)
def load_vectors(path):
    return Sense2Vec().from_disk(path)


st.sidebar.title("sense2vec")
st.sidebar.markdown(
    "Explore semantic similarities of multi-word phrases using "
    "[`sense2vec`](https://github.com/explosion/sense2vec/)."
)

word = st.sidebar.text_input("Word", DEFAULT_WORD)
sense_dropdown = st.sidebar.empty()
n_similar = st.sidebar.slider("Max number of similar entries", 1, 100, value=20, step=1)
show_senses = st.sidebar.checkbox("Distinguish results by sense")
vectors_path = st.sidebar.selectbox("Vectors", SENSE2VEC_PATHS)

if not vectors_path:
    st.error(
        f"""
#### No vectors available
You can pass one or more paths to this
script on the command line. For example:
```bash
streamlit run {sys.argv[0]} /path/to/sense2vec /path/to/other_sense2vec
```
"""
    )
else:
    s2v = load_vectors(vectors_path)
    sense = sense_dropdown.selectbox("Sense", ["auto"] + s2v.senses)
    key = s2v.get_best_sense(word) if sense == "auto" else s2v.make_key(word, sense)
    st.header(f"{word} ({sense})")
    if key is None or key not in s2v:
        st.error(f"**Not found:** No vector available for '{word}' ({sense}).")
    else:
        most_similar = s2v.most_similar(key, n=n_similar)
        rows = []
        seen = set()
        for sim_key, sim_score in most_similar:
            sim_word, sim_sense = s2v.split_key(sim_key)
            if not show_senses and sim_word in seen:
                continue
            seen.add(sim_word)
            sim_freq = s2v.get_freq(sim_key)
            if show_senses:
                sim_word = f"{sim_word} `{sim_sense}`"
            row = f"| {sim_word} | `{sim_score:.3f}` | {sim_freq:,} |"
            rows.append(row)
        table_rows = "\n".join(rows)
        table = f"""
        | Word | Similarity | Frequency |
        | --- | ---: | ---: |
        {table_rows}
        """
        st.markdown(table)
