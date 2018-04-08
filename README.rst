sense2vec: Use NLP to go beyond vanilla word2vec
************************************************

sense2vec (`Trask et. al <https://arxiv.org/abs/1511.06388>`_, 2015) is a nice
twist on `word2vec <https://en.wikipedia.org/wiki/Word2vec>`_ that lets you
learn more interesting, detailed and context-sensitive word vectors. For an
interactive example of the technology, see our
`sense2vec demo <https://demos.explosion.ai/sense2vec>`_ that lets you explore
semantic similarities across all Reddit comments of 2015.

This library is a simple Python/Cython implementation for loading and querying
sense2vec models. While it's best used in combination with
`spaCy <http://spacy.io>`_, the ``sense2vec`` library itself is very lightweight
and can also be used as a standalone module. See below for usage details.

.. image:: https://img.shields.io/travis/explosion/sense2vec/master.svg?style=flat-square
    :target: https://travis-ci.org/explosion/sense2vec
    :alt: Build Status

.. image:: https://img.shields.io/github/release/explosion/sense2vec.svg?style=flat-square
    :target: https://github.com/explosion/sense2vec/releases
    :alt: Current Release Version

.. image:: https://img.shields.io/pypi/v/sense2vec.svg?style=flat-square
    :target: https://pypi.python.org/pypi/sense2vec
    :alt: pypi Version

Usage Examples
==============

Usage with spaCy
----------------

.. code:: python

    import spacy
    from sense2vec import Sense2VecComponent

    nlp = spacy.load('en')
    s2v = Sense2VecComponent('/path/to/reddit_vectors-1.1.0')
    nlp.add_pipe(s2v)

    doc = nlp(u"A sentence about natural language processing.")
    assert doc[3].text == u"natural language processing"
    freq = doc[3]._.s2v_freq
    vector = doc[3]._.s2v_vec
    most_similar = doc[3]._.s2v_most_similar(3)
    # [(('natural language processing', 'NOUN'), 1.0),
    #  (('machine learning', 'NOUN'), 0.8986966609954834),
    #  (('computer vision', 'NOUN'), 0.8636297583580017)]

Standalone usage without spaCy
------------------------------

.. code:: python

    import sense2vec

    s2v = sense2vec.load('/path/to/reddit_vectors-1.1.0')
    query = 'natural_language_processing|NOUN'
    assert query in s2v
    freq, vector = s2v[query]
    words, scores = s2v.most_similar(vector, 3)
    most_similar = list(zip(words, scores))
    # [('natural language processing|NOUN', 1.0),
    #  ('machine learning|NOUN', 0.8986966609954834),
    #  ('computer vision|NOUN', 0.8636297583580017)]

Installation & Setup
====================

==================== ===
**Operating system** macOS / OS X, Linux, Windows (Cygwin, MinGW, Visual Studio)
**Python version**   CPython 2.7, 3.4+. Only 64 bit.
**Package managers** `pip <https://pypi.python.org/pypi/sense2vec>`_ (source packages only)
==================== ===

sense2vec releases are available as source packages on pip:

.. code:: bash

    pip install sense2vec

The Reddit vectors model is attached to the
`latest release <https://github.com/explosion/sense2vec/releases>`_. To load it
in, download the ``.tar.gz`` archive, unpack it and point ``sense2vec.load`` to
the extracted data directory:

.. code:: python

    import sense2vec
    s2v = sense2vec.load('/path/to/reddit_vectors-1.1.0')

Usage
=====

Usage with spaCy v2.x
---------------------

The easiest way to use the library and vectors is to plug it into your spaCy
pipeline. Note that ``sense2vec`` doesn't depend on spaCy, so you'll have to
install it separately and download the English model.

.. code:: bash

    pip install -U spacy
    python -m spacy download en

The ``sense2vec`` package exposes a ``Sense2VecComponent``, which can be
initialised with the data path and added to your spaCy pipeline as a
`custom pipeline component <https://spacy.io/usage/processing-pipelines#custom-components>`_.
By default, components are added to the *end of the pipeline*, which is the
recommended position for this component, since it needs access to the dependency
parse and, if available, named entities.

.. code:: python

    import spacy
    from sense2vec import Sense2VecComponent

    nlp = spacy.load('en')
    s2v = Sense2VecComponent('/path/to/reddit_vectors-1.1.0')
    nlp.add_pipe(s2v)

The pipeline component will **merge noun phrases and entities** according to
the same schema used when training the sense2vec models (e.g. noun chunks
without determiners like "the"). This ensures that you'll be able to retrieve
meaningful vectors for phrases in your text. The component will also add
serveral `extension attributes and methods <https://spacy.io/usage/processing-pipelines#custom-components-attributes>`_
to spaCy's ``Token`` and ``Span`` objects that let you retrieve vectors and
frequencies, as well as most similar terms.

.. code:: python

    doc = nlp(u"A sentence about natural language processing.")
    assert doc[3].text == u"natural language processing"
    assert doc[3]._.in_s2v
    freq = doc[3]._.s2v_freq
    vector = doc[3]._.s2v_vec
    most_similar = doc[3]._.s2v_most_similar(10)

For entities, the entity labels are used as the "sense" (instead of the
token's part-of-speech tag):

.. code:: python

    doc = nlp(u"A sentence about Facebook and Google.")
    for ent in doc.ents:
        assert ent._.in_s2v
        most_similar = ent._.s2v_most_similar(3)

Available attributes
^^^^^^^^^^^^^^^^^^^^

The following attributes are available via the `._` property – for example
``token._.in_s2v``:

==================== ============== ==================== ===
Name                 Attribute Type Type                 Description
==================== ============== ==================== ===
``in_s2v``           property       bool                 Whether a key exists in the vector map.
``s2v_freq``         property       int                  The frequency of the given key.
``s2v_vec``          property       ``ndarray[float32]`` The vector of the given key.
``s2v_most_similar`` method         list                 Get the ``n`` most similar terms. Returns a list of ``((word, sense), score)`` tuples.
==================== ============== ==================== ===

**A note on span attributes:**  Under the hood, entities in ``doc.ents`` are
``Span`` objects. This is why the pipeline component also adds attributes and
methods to spans and not just tokens. However, it's not recommended to use the
sense2vec attributes on arbitrary slices of the document, since the model likely
won't have a key for the respective text. ``Span`` objects also don't have a
part-of-speech tag, so if no entity label is present, the "sense" defaults to
the root's part-of-speech tag.

Standalone usage
----------------

To use only the ``sense2vec`` library, you can import the package and then call
its ``load()`` method to load in the vectors.

.. code:: python

    import sense2vec
    s2v = sense2vec.load('/path/to/reddit_vectors-1.1.0')

``sense2vec.load`` returns an instance of the ``VectorMap`` class, which you
can interact with via the following methods:

``VectorMap.__len__``
^^^^^^^^^^^^^^^^^^^^^

The total number of entries in the map.

=========== ==== ===
Argument    Type Description
=========== ==== ===
**RETURNS** int  The number of entries in the map.
=========== ==== ===

.. code:: python

    s2v = sense2vec.load('/path/to/reddit_vectors-1.1.0')
    assert len(s2v) == 1195261

``VectorMap.__contains__``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Check whether the ``VectorMap`` has a given key. Keys consist of the word
string, a pipe and the "sense", i.e. the part-of-speech tag or entity label.
For example: ``'duck|NOUN'`` or ``'duck|VERB'``. See the section on "Senses"
below for more details. Also note that the underlying vector table is
**case-sensitive**.

=========== ======= ===
Argument    Type    Description
=========== ======= ===
``string``  unicode The key to check.
**RETURNS** bool    Whether the key is part of the map.
=========== ======= ===

.. code:: python

    assert 'duck|NOUN' in s2v
    assert 'duck|VERB' in s2v
    assert 'dkdksl|VERB' not in s2v

``VectorMap.__getitem__``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Retrieve a ``(frequency, vector)`` tuple from the vector map. The frequency is
an integer, the vector a ``numpy.ndarray(dtype='float32')``. If the key is not
found, a ``KeyError`` is raised.

=========== ======= ===
Argument    Type    Description
=========== ======= ===
``string``  unicode The key to retrieve the frequency and vector for.
**RETURNS** tuple   The ``(frequency, vector)`` tuple.
=========== ======= ===

.. code:: python

    freq, vector = s2v['duck|NOUN']

``VectorMap.__setitem__``
^^^^^^^^^^^^^^^^^^^^^^^^^

Assign a ``(frequency, vector)`` tuple to the vector map. The frequency should
be an integer, the vector a ``numpy.ndarray(dtype='float32')``.

=========== ======= ===
Argument    Type    Description
=========== ======= ===
``key``     unicode The key to assign the frequency and vector to.
``value``   tuple   The ``(frequency, vector)`` tuple to assign.
=========== ======= ===

.. code:: python

    freq, vector = s2v['avocado|NOUN']
    s2v['🥑|NOUN'] = (freq, vector)

``VectorMap.__iter__``, ``VectorMap.keys``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Iterate over the keys in the map, in order of insertion.

=========== ======= ===
Argument    Type    Description
=========== ======= ===
**YIELDS**  unicode The keys in the map.
=========== ======= ===

``VectorMap.values``
^^^^^^^^^^^^^^^^^^^^

Iterate over the values in the map, in order of insertion and yield
``(frequency, vector)`` tuples from the vector map. The frequency is an integer,
the vector a ``numpy.ndarray(dtype='float32')``

=========== ======= ===
Argument    Type    Description
=========== ======= ===
**YIELDS**  tuple   The values in the map.
=========== ======= ===

``VectorMap.items``
^^^^^^^^^^^^^^^^^^^

Iterate over the items in the map, in order of insertion and yield
``(key, (frequency, vector))`` tuples from the vector map. The frequency is an integer, the vector a ``numpy.ndarray(dtype='float32')``

=========== ======= ===
Argument    Type    Description
=========== ======= ===
**YIELDS**  tuple   The items in the map.
=========== ======= ===

``VectorMap.most_similar``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Find the keys of the ``n`` most similar entries, given a vector. Note that
the *most* similar entry with a score of ``1.0`` will be the key of the query
vector itself.

=========== ================================== ===
Argument    Type                               Description
=========== ================================== ===
``vector``  ``numpy.ndarray(dtype='float32')`` The vector to compare to.
``n``       int                                The number of entries to return. Defaults to ``10``.
**RETURNS** tuple                              A ``(words, scores)`` tuple.
=========== ================================== ===

.. code:: python

    freq, vector = s2v['avocado|NOUN']
    words, scores = s2v.most_similar(vector, n=3)
    for word, score in zip(words, scores):
        print(word, score)
    # avocado|NOUN 1.0
    # avacado|NOUN 0.970944344997406
    # spinach|NOUN 0.962776780128479

``VectorMap.save``
^^^^^^^^^^^^^^^^^^

Serialize the model to a directory. This will export three files to the output
directory: a  ``strings.json`` containing the keys in insertion order, a
``freqs.json`` containing the frequencies and a ``vectors.bin`` containing the
vectors.

============ ======= ===
Argument     Type    Description
============ ======= ===
``data_dir`` unicode The path to the output directory.
============ ======= ===

``VectorMap.load``
^^^^^^^^^^^^^^^^^^

Load a model from a directory. Expects three files in the directory (see
``VectorMap.save`` for details).

============ ======= ===
Argument     Type    Description
============ ======= ===
``data_dir`` unicode The path to load the model from.
============ ======= ===

Senses
======

The pre-trained Reddit vectors support the following "senses", either
part-of-speech tags or entity labels. For more details, see spaCy's
`annotation scheme overview <https://spacy.io/api/annotation>`_.

========= ========================== ===
Tag       Description                Examples
========= ========================== ===
``ADJ``   adjective                  big, old, green
``ADP``   adposition                 in, to, during
``ADV``   adverb                     very, tomorrow, down, where
``AUX``   auxiliary                  is, has (done), will (do)
``CONJ``  conjunction                and, or, but
``DET``   determiner                 a, an, the
``INTJ``  interjection               psst, ouch, bravo, hello
``NOUN``  noun                       girl, cat, tree, air, beauty
``NUM``   numeral                    1, 2017, one, seventy-seven, MMXIV
``PART``  particle                   's, not
``PRON``  pronoun                    I, you, he, she, myself, somebody
``PROPN`` proper noun                Mary, John, London, NATO, HBO
``PUNCT`` punctuation                , ? ( )
``SCONJ`` subordinating conjunction  if, while, that
``SYM``   symbol                     $, %, =, :), 😝
``VERB``  verb                       run, runs, running, eat, ate, eating
========= ========================== ===

=============== ===
Entity Label    Description
=============== ===
``PERSON``      People, including fictional.
``NORP``        Nationalities or religious or political groups.
``FACILITY``    Buildings, airports, highways, bridges, etc.
``ORG``         Companies, agencies, institutions, etc.
``GPE``         Countries, cities, states.
``LOC``         Non-GPE locations, mountain ranges, bodies of water.
``PRODUCT``     Objects, vehicles, foods, etc. (Not services.)
``EVENT``       Named hurricanes, battles, wars, sports events, etc.
``WORK_OF_ART`` Titles of books, songs, etc.
``LANGUAGE``    Any named language.
=============== ===

Training a sense2vec model
==========================

**🚧 TODO:** Update training scripts for spaCy v2.x.
