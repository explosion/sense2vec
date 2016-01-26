#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Contributed by Matthew Honnibal <matt@spacy.io>
# Copyright (C) 2015 ceded to Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# cython: infer_types=True
from libc.stdint cimport uint64_t, int64_t

from murmurhash.mrmr cimport hash64
from preshed.maps cimport map_get, map_set
from preshed.maps cimport PreshMap
from preshed.counter cimport count_t
from spacy.strings cimport StringStore
from cpython.exc cimport PyErr_CheckSignals


def count_words_fast(StringStore vocab, PreshMap counts, bytes text, int min_freq):
    cdef char* chars = <char*>text
    cdef char space = ' '
    cdef char newline = '\n'
    cdef char c
    cdef int64_t length = len(text)
    cdef int64_t i = 0
    cdef int64_t start = 0
    for i in range(length):
        if chars[i] == space or chars[i] == newline:
            if start < i:
                count_word(vocab, counts, &chars[start], i-start, min_freq)
            start = i+1
    if start < i:
        count_word(vocab, counts, &chars[start], i-start, min_freq)
    PyErr_CheckSignals()


cdef inline int count_word(StringStore vocab, PreshMap counts, char* word,
                            int length, int min_freq) except -1:
    key = hash64(word, length, 1)
    count = <count_t>map_get(counts.c_map, key) + 1
    map_set(counts.mem, counts.c_map, key, <void*>count)
    # Remember the string when we hit min count
    if count == min_freq:
        _ = vocab[word[:length]]

