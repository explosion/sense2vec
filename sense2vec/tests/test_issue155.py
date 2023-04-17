from pathlib import Path
import pytest
from sense2vec.sense2vec import Sense2Vec
from thinc.api import use_ops
from thinc.util import has_cupy_gpu


@pytest.mark.skipif(not has_cupy_gpu, reason="requires Cupy/GPU")
def test_issue155():
    data_path = Path(__file__).parent / "data"
    with use_ops("cupy"):
        s2v = Sense2Vec().from_disk(data_path)
        s2v.most_similar("beekeepers|NOUN")
