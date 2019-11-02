from .sense2vec import Sense2Vec  # noqa: F401
from .component import Sense2VecComponent  # noqa: F401
from .util import importlib_metadata, registry  # noqa: F401

try:
    # This needs to be imported in order for the entry points to be loaded
    from . import prodigy_recipes  # noqa: F401
except ImportError:
    pass

__version__ = importlib_metadata.version(__name__)
