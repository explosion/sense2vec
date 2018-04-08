# coding: utf8
from __future__ import unicode_literals

from .vectors import VectorMap
from .about import __version__


def load(vectors_path):
    vector_map = VectorMap(128)
    vector_map.load(vectors_path)
    return vector_map
