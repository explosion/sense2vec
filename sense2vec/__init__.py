from typing import Union
from pathlib import Path

from .about import __version__  # noqa: F401
from .sense2vec import Sense2Vec  # noqa: F401
from .component import Sense2VecComponent  # noqa: F401


def load(vectors_path: Union[Path, str]) -> Sense2Vec:
    # TODO: remove this?
    if not Path(vectors_path).exists():
        raise IOError(f"Can't find vectors: {vectors_path}")
    return Sense2Vec().from_disk(vectors_path)
