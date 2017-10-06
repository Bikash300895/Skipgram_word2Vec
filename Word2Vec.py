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

