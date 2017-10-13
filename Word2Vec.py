import numpy as np
import tensorflow as tf
import utils
from random import randint
import time

with open('data/text8') as f:
    text = f.read()                 # all sequential text data
    
# Preprocessing the data
# process the raw data, replace systex with text and return a list in sequence of word
words = utils.preprocess(text)

print("Total words: {}".format(len(words)))             # 16,680,599
print("Unique words: {}".format(len(set(words))))       # 63,641


# making a look up table
# vocab_to_int['a'] = 5, which the index of token 'a'
# int_to_vocab[5] = 'a'
vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)

# converting the entire data represented in form of foken number
int_words = [vocab_to_int[word] for word in words]



# Subsampling
from collections import Counter
import random

threshold = 1e-5
number_of_words = len(int_words)
word_counter = Counter(int_words)

frequencies = dict()
drop_probabilities = dict()
train_words = []

droped_count = 0
for word, count in word_counter.items():
    frequency = count/number_of_words
    frequencies[word] = frequency
    drop_probabilities[word] = 1 - np.sqrt(threshold/frequency)

# Keeping the word with word probability < .85    
for word in int_words:
    if drop_probabilities[word] < 0.85:
        train_words.append(word)
        
print(len(train_words))



