from pathlib import Path
import pytest
import spacy
from sense2vec.sense2vec import Sense2Vec
from thinc.util import has_cupy_gpu


@pytest.mark.skipif(not has_cupy_gpu, reason="requires Cupy/GPU")
def test_issue155():
    data_path = Path(__file__).parent / "data"
    spacy.require_gpu()

    s2v = Sense2Vec().from_disk(data_path)
    s2v.most_similar("beekeepers|NOUN")

    # Restore CPU ops for the rest of the session
    spacy.require_cpu()
