#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ngram_loader import NGramLoader

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging
import json
import random
import time

torch.manual_seed(1)

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, context_size, embedding_dim=10):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs



class TextorTrainer(object):

    """ This is the trainer class for Textor
    input params:
    text_path : path to text file . The text file is saved as json lines (jl), each with a key 'text'
    
    context_size : size of previous n-words to for the context for n-gram training
    
    embedding_dim : size of the word embeddings

    """  
    def __init__(self, text_path,  context_size=2 , embedding_dim=10, logger = None):
        
        logging.basicConfig(level=logging.DEBUG)     
        self.__logger = logger or logging.getLogger(__name__)
        self._text_path = text_path        
        self._context_size = context_size
        self._embedding_dim = embedding_dim
        self._words_to_ix = {}
        self._ngram_loader = NGramLoader(self._text_path, self._context_size)
        self._cuda = torch.cuda.is_available()


    def load_words_to_ix(self, words_ix_path):
        with open(words_ix_path):
            self._words_to_ix = json.load(f)

    def train(self, epochs=300):

        if not self._words_to_ix:
            self.words_to_ix()

        losses = []
        loss_function = nn.NLLLoss()
        if self._cuda:
            loss_function = loss_function.cuda()

        model = NGramLanguageModeler(len(self._words_to_ix), self._context_size, self._embedding_dim )
        if self._cuda:
            self.__logger.info("using gpu")
            model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.001)

        self.__logger.info("Training for vocab of size {}...".format(len(self._words_to_ix)))
        
        start_time = time.time()
        for epoch in range(epochs):
            total_loss = torch.Tensor([0])
            if self._cuda:
                total_loss = total_loss.cuda()
            for context_batch, target_batch in self._ngram_loader.ngrams_batch(1, 3000):
                #self.__logger.debug("context : {} , target: {}".format(context, target))
                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in variables)
                context_batch_idxs = []
                
                for context in context_batch:
                    context_idxs = [self._words_to_ix[w] for w in context]
                    context_batch_idxs.append(context_idxs)

                context_var = autograd.Variable(torch.LongTensor(context_batch_idxs))
                
                if self._cuda:
                    context_var = context_var.cuda()

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                model.zero_grad()

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = model(context_var)

                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a variable)
                target_batch_idxs = [self._words_to_ix[target] for target in target_batch]
                target_var = autograd.Variable(torch.LongTensor(target_batch_idxs))

                if self._cuda:
                    target_var = target_var.cuda()
                loss = loss_function(log_probs, target_var)

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                total_loss += loss.data
            self.__logger.info("Epoch {}: loss = {}".format(epoch, int(total_loss)))
            losses.append(total_loss)
        time_elapsed = int((time.time() - start_time)) 
        self.__logger.info("training time for {} epochs  : {} seconds".format(epochs, time_elapsed))
        torch.save(model, 'models/ngram_test.pt')
    
    def words_to_ix(self):

        self.__logger.info("Creating words-to-index lookup")
        count = 0
        for word in self._ngram_loader.words():
            if not word in self._words_to_ix:
                self._words_to_ix[word] = count
                count +=1
        
        with open(self._text_path.split('.')[0]+".json", 'w') as f:
            json.dump(self._words_to_ix, f, sort_keys=True, indent=4)
        
    def __str__(self):
        return "{}({!r})".format(__class__.__name__, self._text_path) 

    def __repr__(self):

        return "{}({!r},{!r},{!r})".format(__class__.__name__, self._text_path, self._context_size, self._embedding_dim)



class TextorPredict(object):

    def __init__(self, model_path, words_ix_path, context_size=2, embedding_dim=10):

        
        self._rev_lookup = self._load_lookup(words_ix_path)
        #self._model = NGramLanguageModeler(len(self._rev_lookup), context_size, embedding_dim)
        self._context_size = context_size
        self._model = torch.load(model_path)
        self._cuda = torch.cuda.is_available()
    
    def predict(self, context = None,  n=1):
    
        if context is None:
            context = []
            for i in range(self._context_size):
                context.append(random.randint(0, len(self._rev_lookup)))
            context = tuple(context)

        context_text = [self._rev_lookup[idx] for idx in context]
        logging.debug("random init : {}".format(context_text))
        for i in range(n):

            context_var = autograd.Variable(torch.LongTensor(context))
            if self._cuda:
                context_var = context_var.cuda()
            pred = torch.max(self._model(context_var), 1)[1]
            pred = int(pred.data.cpu().numpy()[0])
            context_list = list(context)
            context_list.pop(0)
            context_list.append(int(pred))
            context = tuple(context_list)
            context_var = autograd.Variable(torch.LongTensor(context))
            
            yield self._rev_lookup[pred]

    def _load_lookup(self, words_ix_path):
        words_to_ix = None
        with open(words_ix_path) as f:
            words_to_ix = json.load(f)

        rev_lookup = {idx:word for word, idx in words_to_ix.items()}
        return rev_lookup

if __name__=="__main__":

    import sys

    context_size = 3
    textor_trainer = TextorTrainer(sys.argv[1], context_size = context_size)
    #textor_trainer.train(epochs = 800 )

    textor_predictor = TextorPredict('models/ngram_test.pt', 'data/test_book.json', context_size = context_size)

    for text in textor_predictor.predict(n=100):
        print(text , end=" ")

    print("\n")
