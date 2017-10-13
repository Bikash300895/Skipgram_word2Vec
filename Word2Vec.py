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


# Making batches
# function to give context words with window size and center word index given
def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    
    # we will not take exact window size word, rather take a number less than that
    random_count = randint(1,window_size) #the R number described in the description
    
    if idx - random_count < 0:
        start_word = 0
    else:
        start_word = idx - random_count
        
    if idx + random_count > len(words) - 1:
        end_word = len(words)  
    else:
        end_word = idx + random_count + 1
 
    return list(set(words[start_word:idx]+words[idx+1:end_word]))

# print(get_target([0,1,2,3,4,5,6,7,8,9],4,3)) #returns a list of the words around the given index
    

def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    
    n_batches = len(words)//batch_size
    
    # only full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y
