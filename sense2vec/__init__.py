from . import util
# coding: utf8
from __future__ import unicode_literals

from .vectors import VectorMap
from .about import __version__


def load(name=None, via=None):
    package = util.get_package_by_name(name, via=via)
    vector_map = VectorMap(128)
    vector_map.load(package.path)
    return vector_map
