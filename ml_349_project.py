# -*- coding: utf-8 -*-
"""ML 349 - Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d0U9Llp3l-aLDjiaiqf7JevMTe3uW_MW
"""

import re

"""## Read data"""

lines = []
with open('wiki.test.dot', "r") as f:
  lines = f.readlines()

print(len(lines))

"""## Pre-processing"""

def extractWord(line):
    # -- Remove punctuation
    line = re.sub(r'[^\w\s]', '', line)
    # -- Split into words by space
    words = line.split(' ')
    # -- Remove characters
    regex_pattern = r'([^a-zA-Z ]+?)'
    words = [re.sub(regex_pattern, '', x) for x in words]
    for word in words:
        if len(word) < 2 :
            words.remove(word)
    # -- Remove capitalization
    words = [x.lower() for x in words]

    return words

prcoessed_lines = []
words = []
for line in lines:
    if len(line) < 4:
        lines.remove(line)
    else:
        words += extractWord(line)
print('len of lines:', len(lines))
print('number of words:', len(words))
print('examples of processed words:', words[9:20])



"""**NEURAL NETWORK APPROACH**"""

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load document
in_filename = 'wiki.train.dot'
doc = load_doc(in_filename)
print(doc[:200])

import string
 
# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
  doc = doc.replace('--', ' ')
  #replace <unk> with a space
  doc = doc.replace('<unk>', ' ')
	# split into tokens by white space
  tokens = doc.split()
	# remove punctuation from each token
  table = str.maketrans('', '', string.punctuation)
  tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
  tokens = [word for word in tokens if word.isalpha()]
	# make lower case
  tokens = [word.lower() for word in tokens]
  #removing stopwords
  return tokens

# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# save sequences to file
out_filename = 'wiki_train_sequences.txt'
save_doc(sequences, out_filename)

#Training the model

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load
in_filename = 'wiki_train_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

# integer encode sequences of words
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
import numpy as np
from tensorflow.keras.utils import to_categorical
sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]