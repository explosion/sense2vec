# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython.parallel
cimport cpython.array

from libc.stdint cimport int32_t
from libc.stdint cimport uint64_t
from libc.string cimport memcpy
from libc.math cimport sqrt

from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from preshed.maps cimport PreshMap
from murmurhash.mrmr cimport hash64

from cymem.cymem cimport Pool
cimport numpy as np
import numpy
from os import path
try:
    import ujson as json
except ImportError:
    import json

from .cfile cimport CFile
from ._strings cimport StringStore, hash_string


ctypedef pair[float, int] Entry
ctypedef priority_queue[Entry] Queue
ctypedef float (*do_similarity_t)(const float* v1, const float* v2,
        float nrm1, float nrm2, int nr_dim) nogil


cdef struct _CachedResult:
    int* indices
    float* scores
    int n


cdef class VectorMap:
    '''Provide key-based access into the VectorStore. Keys are unicode strings.
    Also manage freqs.'''
    def __init__(self, nr_dim):
        self.data = VectorStore(nr_dim)
        self.strings = StringStore()
        self.freqs = PreshMap()

    @property
    def nr_dim(self):
        return self.data.nr_dim

    def __len__(self):
        '''Number of entries in the map.

        Returns: length int >= 0
        '''
        return self.data.vectors.size()

    def __contains__(self, unicode string):
        '''Check whether the VectorMap has a given key.

        Returns: has_key bool
        '''
        cdef uint64_t hashed = hash_string(string)
        return bool(self.freqs[hashed])

    def __getitem__(self, unicode key):
        '''Retrieve a (frequency, vector) tuple from the vector map, or
        raise KeyError if the key is not found.

        Arguments:
            key unicode

        Returns:
            tuple[int, float32[:self.nr_dim]]
        '''
        cdef uint64_t hashed = hash_string(key)
        freq = self.freqs[hashed]
        if not freq:
            raise KeyError(key)
        else:
            i = self.strings[key]
            return freq, self.data[i]

    def __setitem__(self, unicode key, value):
        '''Assign a (frequency, vector) tuple to the vector map.

        Arguments:
            key unicode
            value tuple[int, float32[:self.nr_dim]]
        Returns:
            None
        '''
        # TODO: Handle case where we're over-writing an existing entry.
        cdef int freq
        cdef float[:] vector
        freq, vector = value
        idx = self.strings[key]
        cdef uint64_t hashed = hash_string(key)
        self.freqs[hashed] = freq
        assert self.data.vectors.size() == idx
        self.data.add(vector)

    def __iter__(self):
        '''Iterate over the keys in the map, in order of insertion.

        Generates:
            key unicode
        '''
        yield from self.strings

    def keys(self):
        '''Iterate over the keys in the map, in order of insertion.

        Generates:
            key unicode
        '''
        yield from self.strings

    def values(self):
        '''Iterate over the values in the map, in order of insertion.

        Generates:
            (freq,vector) tuple[int, float32[:self.nr_dim]]
        '''
        for key, value in self.items():
            yield value

    def items(self):
        '''Iterate over the items in the map, in order of insertion.

        Generates:
            (key, (freq,vector)): tuple[int, float32[:self.nr_dim]]
        '''
        cdef uint64_t hashed
        for i, string in enumerate(self.strings):
            hashed = hash_string(string)
            freq = self.freqs[hashed]
            yield string, (freq, self.data[i])


    def most_similar(self, float[:] vector, int n=10):
        '''Find the keys of the N most similar entries, given a vector.

        Arguments:
            vector float[:]
            n int default=10

        Returns:
            list[unicode] length<=n
        '''
        indices, scores = self.data.most_similar(vector, n)
        return [self.strings[idx] for idx in indices], scores

    def add(self, unicode string, int freq, float[:] vector):
        '''Insert a vector into the map by value. Makes a copy of the vector.
        '''
        idx = self.strings[string]
        cdef uint64_t hashed = hash_string(string)
        self.freqs[hashed] = freq
        assert self.data.vectors.size() == idx
        self.data.add(vector)

    def borrow(self, unicode string, int freq, float[:] vector):
        '''Insert a vector into the map by reference. Does not copy the data, and
        changes to the vector will be reflected in the VectorMap.

        The user is responsible for ensuring that another reference to the vector
        is maintained --- otherwise, the Python interpreter will free the memory,
        potentially resulting in an invalid read.
        '''
        idx = self.strings[string]
        cdef uint64_t hashed = hash_string(string)
        self.freqs[hashed] = freq
        assert self.data.vectors.size() == idx
        self.data.borrow(vector)

    def save(self, data_dir):
        '''Serialize to a directory.

        * data_dir/strings.json --- The keys, in insertion order.
        * data_dir/freqs.json --- The frequencies.
        * data_dir/vectors.bin --- The vectors.
        '''
        self.strings.to_disk(path.join(data_dir, 'strings.json'))
        self.data.save(path.join(data_dir, 'vectors.bin'))
        freqs = []
        cdef uint64_t hashed
        for string in self.strings:
            hashed = hash_string(string)
            freq = self.freqs[hashed]
            if not freq:
                continue
            freqs.append([string, freq])
        with open(path.join(data_dir, 'freqs.json'), 'w') as file_:
            json.dump(freqs, file_)

    def load(self, data_dir):
        '''Load from a directory:

        * data_dir/strings.json --- The keys, in insertion order.
        * data_dir/freqs.json --- The frequencies.
        * data_dir/vectors.bin --- The vectors.
        '''
        self.data.load(path.join(data_dir, 'vectors.bin'))
        self.strings.from_disk(path.join(data_dir, 'strings.json'))
        with open(path.join(data_dir, 'freqs.json')) as file_:
            freqs = json.load(file_)
        cdef uint64_t hashed
        for string, freq in freqs:
            hashed = hash_string(string)
            self.freqs[hashed] = freq


cdef class VectorStore:
    '''Maintain an array of float* pointers for word vectors, which the
    table may or may not own. Keys and frequencies sold separately ---
    we're just a dumb vector of data, that knows how to run linear-scan
    similarity queries.'''
    def __init__(self, int nr_dim):
        self.mem = Pool()
        self.nr_dim = nr_dim
        zeros = <float*>self.mem.alloc(self.nr_dim, sizeof(float))
        self.vectors.push_back(zeros)
        self.norms.push_back(0)
        self.cache = PreshMap(100000)

    def __getitem__(self, int i):
        cdef float* ptr = self.vectors.at(i)
        cv = <float[:self.nr_dim]>ptr
        return numpy.asarray(cv, dtype='float32')

    def add(self, float[:] vec):
        assert len(vec) == self.nr_dim
        ptr = <float*>self.mem.alloc(self.nr_dim, sizeof(float))
        memcpy(ptr,
            &vec[0], sizeof(ptr[0]) * self.nr_dim)
        self.norms.push_back(get_l2_norm(&ptr[0], self.nr_dim))
        self.vectors.push_back(ptr)

    def borrow(self, float[:] vec):
        self.norms.push_back(get_l2_norm(&vec[0], self.nr_dim))
        # Danger! User must ensure this is memory contiguous!
        self.vectors.push_back(&vec[0])

    def similarity(self, float[:] v1, float[:] v2):
        '''Measure the similarity between two vectors, using cosine.

        Arguments:
            v1 float[:]
            v2 float[:]

        Returns:
            similarity_score -1<float<=1
        '''
        norm1 = get_l2_norm(&v1[0], len(v1))
        norm2 = get_l2_norm(&v2[0], len(v2))
        return cosine_similarity(&v1[0], &v2[0], norm1, norm2, len(v1))

    def most_similar(self, float[:] query, int n):
        cdef int[:] indices = np.ndarray(shape=(n,), dtype='int32')
        cdef float[:] scores = np.ndarray(shape=(n,), dtype='float32')
        cdef uint64_t cache_key = hash64(&query[0], sizeof(query[0]) * n, 0)
        cached_result = <_CachedResult*>self.cache.get(cache_key)
        if cached_result is not NULL and cached_result.n == n:
            memcpy(&indices[0], cached_result.indices, sizeof(indices[0]) * n)
            memcpy(&scores[0], cached_result.scores, sizeof(scores[0]) * n)
        else:
            # This shouldn't happen. But handle it if it does
            if cached_result is not NULL:
                if cached_result.indices is not NULL:
                    self.mem.free(cached_result.indices)
                if cached_result.scores is not NULL:
                    self.mem.free(cached_result.scores)
                self.mem.free(cached_result)
            self._similarities.resize(self.vectors.size())
            linear_similarity(&indices[0], &scores[0], &self._similarities[0],
                n, &query[0], self.nr_dim,
                &self.vectors[0], &self.norms[0], self.vectors.size(),
                cosine_similarity)
            cached_result = <_CachedResult*>self.mem.alloc(sizeof(_CachedResult), 1)
            cached_result.n = n
            cached_result.indices = <int*>self.mem.alloc(
                sizeof(cached_result.indices[0]), n)
            cached_result.scores = <float*>self.mem.alloc(
                sizeof(cached_result.scores[0]), n)
            self.cache.set(cache_key, cached_result)
            memcpy(cached_result.indices, &indices[0], sizeof(indices[0]) * n)
            memcpy(cached_result.scores, &scores[0], sizeof(scores[0]) * n)
        return indices, scores

    def save(self, loc):
        cdef CFile cfile = CFile(loc, 'wb')
        cdef float* vec
        cdef int32_t nr_vector = self.vectors.size()
        cfile.write_from(&nr_vector, 1, sizeof(nr_vector))
        cfile.write_from(&self.nr_dim, 1, sizeof(self.nr_dim))
        for vec in self.vectors:
            cfile.write_from(vec, self.nr_dim, sizeof(vec[0]))
        cfile.close()

    def load(self, loc):
        cdef CFile cfile = CFile(loc, 'rb')
        cdef int32_t nr_vector
        cfile.read_into(&nr_vector, 1, sizeof(nr_vector))
        cfile.read_into(&self.nr_dim, 1, sizeof(self.nr_dim))
        cdef vector[float] tmp
        tmp.resize(self.nr_dim)
        cdef float[:] cv
        for i in range(nr_vector):
            cfile.read_into(&tmp[0], self.nr_dim, sizeof(tmp[0]))
            ptr = &tmp[0]
            cv = <float[:128]>ptr
            if i >= 1:
                self.add(cv)
        cfile.close()


cdef void linear_similarity(int* indices, float* scores, float* tmp,
        int nr_out, const float* query, int nr_dim,
        const float* const* vectors, const float* norms, int nr_vector,
        do_similarity_t get_similarity) nogil:
    query_norm = get_l2_norm(query, nr_dim)
    # Initialize the partially sorted heap
    cdef int i
    cdef float score
    for i in cython.parallel.prange(nr_vector, nogil=True):
        tmp[i] = get_similarity(query, vectors[i], query_norm, norms[i], nr_dim)
    cdef priority_queue[pair[float, int]] queue
    cdef float cutoff = 0
    for i in range(nr_vector):
        score = tmp[i]
        if score > cutoff:
            queue.push(pair[float, int](-score, i))
            cutoff = -queue.top().first
            if queue.size() > nr_out:
                queue.pop()
    # Fill the outputs
    i = 0
    while i < nr_out and not queue.empty():
        entry = queue.top()
        scores[nr_out-(i+1)] = -entry.first
        indices[nr_out-(i+1)] = entry.second
        queue.pop()
        i += 1


cdef float get_l2_norm(const float* vec, int n) nogil:
    norm = 0.0
    for i in range(n):
        norm += vec[i] ** 2
    return sqrt(norm)


cdef float cosine_similarity(const float* v1, const float* v2,
        float norm1, float norm2, int n) nogil:
    dot = 0.0
    for i in range(n):
        dot += v1[i] * v2[i]
    return dot / (norm1 * norm2)
