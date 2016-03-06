import pytest

import sense2vec


@pytest.mark.models
def test_sample():
    s2v = sense2vec.load('reddit_vectors')
    freq, query_vector = s2v[u"beekeepers|NOUN"]
    assert freq is not None
    assert s2v.most_similar(query_vector, 3)[0] == \
        [u'beekeepers|NOUN', u'honey_bees|NOUN', u'Beekeepers|NOUN']
