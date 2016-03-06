from . import util
from .vectors import VectorMap


def load(name=None, via=None):
    package = util.get_package_by_name(name, via=via)
    vector_map = VectorMap(128)
    vector_map.load(package.path)
    return vector_map
