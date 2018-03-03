#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json_lines as jl
import os
import re
class NGramLoader(object):

    def __init__(self, path, ngram=3):
        assert(os.path.exists(path))
        self._path = path
        self._n = ngram 

    @property
    def n(self):
        return self._n  
    
    def ngrams(self):
        with open(self._path) as f:
            for para in jl.reader(f):
                para = para['text']
                list_words = self._para_to_words(para)

                for i in range(len(list_words)-self._n):
                    context = list_words[i:i+self._n]
                    target =  list_words[i+ self._n]
                    yield context, target

    def _para_to_words(self, para):
        '''
            converts a string to list of words
        '''
        words = re.findall(r"[\w]+", para)
        return words

    def words(self):
        with open(self._path) as f:
            for para in jl.reader(f):
                para = para['text']
                list_words = self._para_to_words(para)

                for word in list_words:
                    yield word


    def __repr__(self):
        return "{}('{}')".format(__class__.__name__, self._path)
