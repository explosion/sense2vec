# coding: utf8
from __future__ import unicode_literals

import pytest
from os import path

from .. import load


data_path = path.join(path.dirname(__file__), '..', '..', 'data')


@pytest.mark.models
@pytest.mark.parametrize('model', ['reddit_vectors-1.1.0'])
def test_sample(model):
    s2v = load(path.join(data_path, model))
    freq, query_vector = s2v[u"beekeepers|NOUN"]
    assert freq is not None
    assert s2v.most_similar(query_vector, 3)[0] == \
        [u'beekeepers|NOUN', u'honey_bees|NOUN', u'Beekeepers|NOUN']
