# sense2vec

Use spaCy to go beyond vanilla word2vec

Read about sense2vec here:

https://spacy.io/blog/sense2vec-with-spacy

You can use an online demo of the technology here:

https://sense2vec.spacy.io/

We're currently refining the API, to make this technology easy to use. Once we've completed that, you'll be able
to download the package on PyPi. For now, the code is available to clarify the blog post.

There are three relevant files in this repository:

# bin/merge_text.py

This script pre-processes text using spaCy, so that the sense2vec model can be trained using Gensim.

# bin/train_word2vec.py

This script reads a directory of text files, and then trains a word2vec model using Gensim. The script includes its own
vocabulary counting code, because Gensim's vocabulary count is a bit slow for our large, sparse vocabulary.

# sense2vec/vectors.pyx

To serve the similarity queries, we wrote a small vector-store class in Cython. This made it easier to add an efficient
cache in front of the service. It also less memory than Gensim's Word2Vec class, as it doesn't hold the keys as Python
unicode strings.

Similarity queries could be faster, if we had made all vectors contiguous in memory, instead of holding them
as an array of pointers. However, we wanted to allow a `.borrow()` method, so that vectors can be added to the store
by reference, without copying the data.

# Downloading the model

The easiest way to download and install the model is by calling python -m sense2vec.download after installing sense2vec, e.g., via pip install -e git+git://github.com/spacy-io/sense2vec.git#egg=sense2vec. Please note that you'll need Blas/Atlas packages installed. On RedHad those are atlas and atlas-devel. You can then load the model as follows:

    import sputnik
    from sense2vec import about
    from sense2vec.vectors import VectorMap

    package = sputnik.package(about.__title__, about.__version__, about.__default_model__)
    vector_map = VectorMap(128)
    vector_map.load(package.path)

# Use on OSX

The code in the master branch was optimized for our Ubuntu servers, and doesn't support OSX, due to usage of OpenMP, and details about which BLAS library to link against.

The `osx` branch contains a version of the library without these optimizations, that compiles on OSX. Full support for OSX and Windows platforms will be available when we release the package on PyPi.
