import numpy as np
import tensorflow as tf
import utils
from random import randint
import time

with open('data/text8') as f:
    text = f.read()
    
# Preprocessing the data
words = utils.preprocess(text)

print("Total words: {}".format(len(words)))
print("Unique words: {}".format(len(set(words))))


# making a look up table
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
    
for word in int_words:
    if drop_probabilities[word] < 0.85:
        train_words.append(word)
        
print(len(train_words))
