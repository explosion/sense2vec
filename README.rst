sense2vec: Use spaCy to go beyond vanilla word2vec
**************************************************

Read about sense2vec in our `blog post <https://spacy.io/blog/sense2vec-with-spacy>`_. You can try an online demo of the technology `here <https://demos.explosion.ai/sense2vec>`_ and use the open-source `REST server <https://github.com/explosion/spacy-services>`_. 

.. image:: https://travis-ci.org/spacy-io/sense2vec.svg?branch=master
    :target: https://travis-ci.org/spacy-io/sense2vec

Overview
========

There are three relevant files in this repository:

``bin/merge_text.py``
-----------------

This script pre-processes text using spaCy, so that the sense2vec model can be trained using Gensim.

``bin/train_word2vec.py``
---------------------

This script reads a directory of text files, and then trains a word2vec model using Gensim. The script includes its own
vocabulary counting code, because Gensim's vocabulary count is a bit slow for our large, sparse vocabulary.

``sense2vec/vectors.pyx``
---------------------

To serve the similarity queries, we wrote a small vector-store class in Cython. This made it easier to add an efficient
cache in front of the service. It also less memory than Gensim's Word2Vec class, as it doesn't hold the keys as Python
unicode strings.

Similarity queries could be faster, if we had made all vectors contiguous in memory, instead of holding them
as an array of pointers. However, we wanted to allow a ``.borrow()`` method, so that vectors can be added to the store
by reference, without copying the data.

Installation
============

Until there is a PyPI release you can install sense2vec by: 

1. cloning the repository 
2. run ``pip install -r requirements.txt``
3. ``pip install -e .``
4. install the latest model via ``sputnik --name sense2vec --repository-url http://index.spacy.io install reddit_vectors``

You might also be tempted to simply run ``pip install -e git+git://github.com/spacy-io/sense2vec.git#egg=sense2vec`` instead of steps 1-3, but it expects `Cython <http://cython.org/>`_ to be present.

Usage
=====

.. code:: python

 import sense2vec
 model = sense2vec.load()
 freq, query_vector = model["natural_language_processing|NOUN"]
 model.most_similar(query_vector, n=3)

.. code:: python

 (['natural_language_processing|NOUN', 'machine_learning|NOUN', 'computer_vision|NOUN'], <MemoryView of 'ndarray'>)

For additional performance experimental support for BLAS can be enabled by setting the `USE_BLAS` environment variable before installing (e.g. ``USE_BLAS=1 pip install ...``). This requires an up-to-date BLAS/OpenBlas/Atlas installation.

Support
=======

* CPython 2.6, 2.7, 3.3, 3.4, 3.5 (only 64 bit)
* OSX
* Linux
* Windows
