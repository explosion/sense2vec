#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Contributed by Matthew Honnibal <matt@spacy.io>
# Copyright (C) 2015 ceded to Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


from cpython cimport PyUnicode_AS_DATA
from cpython cimport PyUnicode_GET_DATA_SIZE
from libc.stdint cimport uint64_t

from murmurhash.mrmr cimport hash64
from preshed.maps cimport map_get, map_set
from preshed.maps cimport PreshMap
from preshed.counter cimport PreshCounter, count_t
from spacy.strings cimport StringStore

from collections import defaultdict
from six import iteritems


cpdef uint64_t _hash_string(unicode string) except 0:
    # This code is copied from spacy.strings. The implementation took some thought,
    # and consultation with Stefan Behnel. Do not change blindly. Interaction
    # with Python 2/3 is subtle.
    chars = <char*>PyUnicode_AS_DATA(string)
    size = PyUnicode_GET_DATA_SIZE(string)
    return hash64(chars, size, 1)


cpdef uint64_t _hash_bytes(bytes string) except 0:
    chars = <char*>string
    return hash64(chars, len(string), 1)


def count_words_fast(sentences, count_t min_freq):
    strings = StringStore()
    sentence_no = -1
    total_words = 0
    cdef uint64_t key
    cdef count_t count
    cdef unicode word
    cdef PreshMap counts = PreshMap()
    for sentence_no, sentence in enumerate(sentences):
        for word in sentence:
            key = _hash_string(word)

            count = <count_t>map_get(counts.c_map, key) + 1
            map_set(counts.mem, counts.c_map, key, <void*>count)
            # Remember the string when we hit min count
            if count == min_freq:
                _ = strings[word]
        total_words += len(sentence)
    
    # Use defaultdict to match the pure Python version of the function
    vocab = defaultdict(int)
    for word in strings:
        key = _hash_string(word)
        vocab[word] = counts[key]
    return vocab, sentence_no

