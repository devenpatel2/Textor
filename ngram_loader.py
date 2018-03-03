#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json_lines as jl
import os
import re
import itertools
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
    
    def ngrams_batch(self, batch_size=10, batches = None):
        batch_idx = 0


        for j in itertools.count(1):
            if batches and j> batches:
                raise StopIteration
            batch_context = []
            batch_target = []
            for context, target in itertools.islice(self.ngrams(), batch_idx, batch_idx + batch_size):
                batch_context.append(context)
                batch_target.append(target)
            
            batch_idx += batch_size

            if len(batch_context) == 0:
                raise StopIteration
            yield batch_context, batch_target

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
