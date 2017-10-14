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


# defining the graph
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32,shape=[None],name = "inputs")
    labels = tf.placeholder(tf.int32,shape = [None,1],name = "labels")
    
    
n_vocab = len(int_to_vocab)
n_embedding =  300 
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform([n_vocab,n_embedding],-1,-1 ))
    embed = tf.nn.embedding_lookup(embedding,inputs)


# Number of negative labels to sample
n_sampled = 100
with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal( [n_vocab,n_embedding],stddev=0.1),name = "softmax_w")
    softmax_b = tf.Variable( tf.zeros([n_vocab]),name = "softmax_b")
    
    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(softmax_w,softmax_b,labels,embed,n_sampled,n_vocab)
    
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)   
    
    
    
    
# Validation
with train_graph.as_default():
    ## From Thushan Ganegedara's implementation
    valid_size = 16 # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, 
                               random.sample(range(1000,1000+valid_window), valid_size//2))

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
    

# Training
epochs = 10
batch_size = 1000
window_size = 10

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs+1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches:
            
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            
            loss += train_loss
            
            if iteration % 100 == 0: 
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/100),
                      "{:.4f} sec/batch".format((end-start)/100))
                loss = 0
                start = time.time()
            
            if iteration % 1000 == 0:
                ## From Thushan Ganegedara's implementation
                # note that this is expensive (~20% slowdown if computed every 500 steps)
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
            
            iteration += 1
    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)


with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    embed_mat = sess.run(embedding)
    

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE    
    
    
    
    
