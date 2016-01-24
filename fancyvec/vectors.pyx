# cython: profile=True
# cython: cdivision=True
# cython: infer_types=True
from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t
from libc.string cimport memcpy
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from spacy.cfile cimport CFile

from cymem.cymem cimport Pool
cimport numpy as np


ctypedef pair[float, int] Entry
ctypedef priority_queue[Entry] Queue
ctypedef float (*do_similarity_t)(const float* v1, const float* v2,
        float nrm1, float nrm2, int nr_dim) nogil


cdef class Vectors:
    cdef readonly Pool mem
    cdef vector[float*] vectors
    cdef vector[float] norms
    cdef readonly int nr_dim
    
    def __init__(self, int nr_dim):
        self.mem = Pool()
        self.nr_dim = nr_dim 

    def add(self, float[:] vec):
        assert len(vec) == self.nr_dim
        ptr = <float*>self.mem.alloc(self.nr_dim, sizeof(float))
        memcpy(ptr,
            &vec[0], sizeof(ptr[0]) * self.nr_dim)
        self.vectors.push_back(ptr)
        self.norms.push_back(get_l2_norm(ptr, self.nr_dim))
    
    def borrow(self, float[:] vec):
        assert len(vec) == self.nr_dim
        # Danger! User must ensure this is memory contiguous!
        self.vectors.push_back(&vec[0])
        self.norms.push_back(get_l2_norm(&vec[0], self.nr_dim))

    def most_similar(self, float[:] query, int n):
        cdef int[:] indices = np.ndarray(shape=(n,), dtype='int32')
        cdef float[:] scores = np.ndarray(shape=(n,), dtype='float32')
        linear_similarity(&indices[0], &scores[0],
            n, &query[0], self.nr_dim,
            &self.vectors[0], &self.norms[0], self.vectors.size(), 
            cosine_similarity)
        return indices, scores

    def save(self, loc):
        cdef CFile cfile = CFile(loc, 'w')
        cdef const float* vec
        for vec in self.vectors:
            cfile.write_from(vec, self.nr_dim, sizeof(vec[0]))
        cfile.close()


cdef void linear_similarity(int* indices, float* scores,
        int nr_out, const float* query, int nr_dim,
        const float* const* vectors, const float* norms, int nr_vector,
        do_similarity_t get_similarity) nogil:
    query_norm = get_l2_norm(query, nr_dim)
    # Initialize the partially sorted heap
    cdef priority_queue[pair[float, int]] queue
    for i in range(nr_out):
        score = get_similarity(query, vectors[i], query_norm, norms[i], nr_dim)
        with gil:
            print("Score", i, score)
        queue.push(pair[float, int](-score, i))
    # Get the rest of the similarities, maintaining the top N
    for i in range(nr_out, nr_vector):
        score = get_similarity(query, vectors[i], query_norm, norms[i], nr_dim)
        if score >= -queue.top().first:
            queue.pop()
            queue.push(pair[float, int](-score, i))
    # Fill the outputs
    i = 0
    while i < nr_out and not queue.empty(): 
        entry = queue.top()
        scores[nr_out-(i+1)] = -entry.first
        indices[nr_out-(i+1)] = entry.second
        queue.pop()
        i += 1
    

cdef extern from "cblas.h":
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
    return dot / ((norm1 * norm2) + 1e-8)
