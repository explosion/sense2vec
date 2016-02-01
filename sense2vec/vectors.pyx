# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
cimport cython.parallel
from libc.stdint cimport int32_t
from libc.stdint cimport uint64_t
from libc.string cimport memcpy
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from spacy.cfile cimport CFile
from preshed.maps cimport PreshMap
from spacy.strings cimport StringStore, hash_string

from cymem.cymem cimport Pool
cimport numpy as np
import numpy
from os import path
import ujson as json


ctypedef pair[float, int] Entry
ctypedef priority_queue[Entry] Queue
ctypedef float (*do_similarity_t)(const float* v1, const float* v2,
        float nrm1, float nrm2, int nr_dim) nogil


cdef class VectorMap:
    '''Provide key-based access into the VectorStore. Keys are unicode strings.
    Also manage freqs.'''
    cdef readonly Pool mem
    cdef VectorStore data
    cdef readonly StringStore strings
    cdef PreshMap freqs
    
    def __init__(self, nr_dim):
        self.data = VectorStore(nr_dim)
        self.strings = StringStore()
        self.freqs = PreshMap()
    
    def __contains__(self, unicode string):
        cdef uint64_t hashed = hash_string(string)
        return bool(self.freqs[hashed])

    def __getitem__(self, unicode string):
        cdef uint64_t hashed = hash_string(string)
        freq = self.freqs[hashed]
        if not freq:
            raise KeyError(string)
        else:
            i = self.strings[string]
            return freq, self.data[i]

    def __iter__(self):
        cdef uint64_t hashed
        for i, string in enumerate(self.strings):
            hashed = hash_string(string)
            freq = self.freqs[hashed]
            yield (string, freq, self.data[i])

    def most_similar(self, float[:] vector, int n):
        indices, scores = self.data.most_similar(vector, n)
        return [self.strings[idx] for idx in indices], scores

    def add(self, unicode string, int freq, float[:] vector):
        idx = self.strings[string]
        cdef uint64_t hashed = hash_string(string)
        self.freqs[hashed] = freq
        assert self.data.vectors.size() == idx
        self.data.add(vector)

    def borrow(self, unicode string, int freq, float[:] vector):
        idx = self.strings[string]
        cdef uint64_t hashed = hash_string(string)
        self.freqs[hashed] = freq
        assert self.data.vectors.size() == idx
        self.data.borrow(vector)

    def save(self, data_dir):
        with open(path.join(data_dir, 'strings.json'), 'w') as file_:
            self.strings.dump(file_)
        self.data.save(path.join(data_dir, 'vectors.bin'))
        freqs = []
        cdef uint64_t hashed
        for hashed, freq in self.freqs.items():
            freqs.append([hashed, freq])
        with open(path.join(data_dir, 'freqs.json'), 'w') as file_:
            json.dump(freqs, file_)

    def load(self, data_dir):
        self.data.load(path.join(data_dir, 'vectors.bin'))
        with open(path.join(data_dir, 'strings.json')) as file_:
            self.strings.load(file_)
        with open(path.join(data_dir, 'freqs.json')) as file_:
            freqs = json.load(file_)
        cdef uint64_t hashed
        for hashed, freq in freqs:
            self.freqs[hashed] = freq


cdef class VectorStore:
    '''Maintain an array of float* pointers for word vectors, which the
    table may or may not own. Keys and frequencies sold separately --- 
    we're just a dumb vector of data, that knows how to run linear-scan
    similarity queries.'''
    cdef readonly Pool mem
    cdef vector[float*] vectors
    cdef vector[float] norms
    cdef vector[float] _similarities
    cdef readonly int nr_dim
    
    def __init__(self, int nr_dim):
        self.mem = Pool()
        self.nr_dim = nr_dim 
        zeros = <float*>self.mem.alloc(self.nr_dim, sizeof(float))
        self.vectors.push_back(zeros)
        self.norms.push_back(0)

    def __getitem__(self, int i):
        cdef float* ptr = self.vectors.at(i)
        cv = <float[:self.nr_dim]>ptr
        return numpy.asarray(cv)

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

    def most_similar(self, float[:] query, int n):
        cdef int[:] indices = np.ndarray(shape=(n,), dtype='int32')
        cdef float[:] scores = np.ndarray(shape=(n,), dtype='float32')
        self._similarities.reserve(self.vectors.size())
        linear_similarity(&indices[0], &scores[0], &self._similarities[0],
            n, &query[0], self.nr_dim,
            &self.vectors[0], &self.norms[0], self.vectors.size(), 
            cosine_similarity)
        return indices, scores

    def save(self, loc):
        cdef CFile cfile = CFile(loc, 'w')
        cdef float* vec
        cdef int32_t nr_vector = self.vectors.size()
        cfile.write_from(&nr_vector, 1, sizeof(nr_vector))
        cfile.write_from(&self.nr_dim, 1, sizeof(self.nr_dim))
        for vec in self.vectors:
            cfile.write_from(vec, self.nr_dim, sizeof(vec[0]))
        cfile.close()

    def load(self, loc):
        cdef CFile cfile = CFile(loc, 'r')
        cdef int32_t nr_vector
        cfile.read_into(&nr_vector, 1, sizeof(nr_vector))
        cfile.read_into(&self.nr_dim, 1, sizeof(self.nr_dim))
        cdef vector[float] tmp
        tmp.reserve(self.nr_dim)
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
        #tmp[i] = cblas_sdot(nr_dim, query, 1, vectors[i], 1) / (query_norm * norms[i])
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
    

cdef extern from "cblas_shim.h":
    float cblas_sdot(int N, float  *x, int incX, float  *y, int incY ) nogil
    float cblas_snrm2(int N, float  *x, int incX) nogil


cdef extern from "math.h" nogil:
    float sqrtf(float x)


DEF USE_BLAS = True


cdef float get_l2_norm(const float* vec, int n) nogil:
    cdef float norm
    if USE_BLAS:
        return cblas_snrm2(n, vec, 1)
    else:
        norm = 0
        for i in range(n):
            norm += vec[i] ** 2
        return sqrtf(norm)


cdef float cosine_similarity(const float* v1, const float* v2,
        float norm1, float norm2, int n) nogil:
    cdef float dot
    if USE_BLAS:
        dot = cblas_sdot(n, v1, 1, v2, 1)
    else:
        dot = 0
        for i in range(n):
            dot += v1[i] * v2[i]
    return dot / (norm1 * norm2)
