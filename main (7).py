import sys
import string
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import re

# Our ReLU based Feedforward NeuralNet
class ReLUbasedFFNeuralNet(nn.Module):
  def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
      super(ReLUbasedFFNeuralNet, self).__init__()
      self.layers = nn.Sequential(
          nn.Linear(in_dim, hidden_dim1),
          nn.ReLU(),
          nn.Linear(hidden_dim1, hidden_dim2),
          nn.ReLU(),
          nn.Linear(hidden_dim2, out_dim)
      )
  
  def forward(self, x):
      return self.layers(x)

# dict <str, number_assigned>
def validchar():
  # pytorch tensor cannot handle string as tensor, so we use ascii code
  chardist = { ".": 0, ",": 1, "=": 2, "?": 3, ";": 4, "!": 5, ":": 6, "'": 7, "\"": 8 }
  i = 0
  for c in string.ascii_lowercase:
    chardist[c] = i + 9
    i += 1
  return chardist

def compute_ve(a_char):
  #torch.use_deterministic_algorithms(True)
  #a_char_idx = ord(a_char)
  #torch.manual_seed(a_char_idx)
  return torch.randn(2240, requires_grad=True)

def probability(y, idx, word, char, charset):
  ans = [0]*64
  for i, ve in enumerate(y):
    #print("i=", i)
    t = [0]*35
    t[charset[char]] = 1
    #ans[i] = torch.dot(ve.detach(), torch.Tensor(t))
    ans[i] = torch.nn.functional.cosine_similarity(ve.detach(), torch.Tensor(t), dim=0)
  v = torch.autograd.Variable(torch.Tensor(ans), requires_grad=True)
  
  # we need to implement the next charactor probability here
  if (idx + 1) < len(word):
    next_char = word[idx + 1]
    ans = [0]*35
    if next_char in charset.keys():
      for i, ve in enumerate(y):
        t = [0]*35
        t[charset[next_char]] = 1
        #ans[i] = torch.dot(ve.detach(), torch.Tensor(t))
        ans[i] = torch.nn.functional.cosine_similarity(ve.detach(), torch.Tensor(t), dim=0)
    n_v = torch.autograd.Variable(torch.Tensor(ans), requires_grad=True)
    return torch.nn.functional.softmax(n_v / (n_v + v), dim=0)
  else:
    return torch.nn.functional.softmax(v, dim=0)

# our training loop
def train_loop(nn, lossfn, optimizer, epochs, train_dataset, valid_dataset, valid_label):
  itr = 0
  accuracy_compute = 10000
  charset = validchar()
  for epoch in range(epochs):
    for i, word in enumerate(train_dataset):
      #if i > 100000:
       # break
      #print("train word = ", word)
      for j, char in enumerate(word):
        # clear gradients with regards to nn parameters on every epoch run
        optimizer.zero_grad()

        # compute the vector embedding before passing to the nn
        if char in charset.keys():
          char_vector_embedding = compute_ve(char)
          #print("y=",char_vector_embedding, " len=", char_vector_embedding.size(dim=0))
          y_ve = nn(char_vector_embedding)
          #print("y_ve=", y_ve)
          label = charset[char]
          l = torch.Tensor([label])
          effective_y = probability(y_ve, j, word, char, charset)
          #print(effective_y.unsqueeze(0))
          # compute our loss
          loss = lossfn(effective_y.unsqueeze(0), l.long())
          #print("loss=", loss)
          loss.backward()
          # update our nn parameters
          optimizer.step()

      itr +=1

      if itr % accuracy_compute == 0:
        # compute accuracy
        correct = 0
        total = 0

        for k, word in enumerate(valid_dataset):
          #if k > 100:
          #  break
          #print("validate word = ", word)
          for j, char in enumerate(word):
            if ord(char) != 183:
              continue
            o_ve = nn(compute_ve(char))
            # to be done later .. we need a map of vector embedding for 
            # each character of label as map for right-wrong judgement here
            #print("o_ve=", o_ve)
            predicted_char = torch.argmax(o_ve)
            #print("k=",k,",j=",j,"v=",valid_label[k][j])
            if k < len(valid_label):
              if j < len(valid_label[k]):
                if valid_label[k][j] in charset.keys():
                  if predicted_char == charset[valid_label[k][j]]:
                      correct += 1
                  total += 1
            
        accuracy = 100 * correct / total
        print('Iteration: ', itr, ' Accuracy: ', accuracy)

# our main training entry point
def train(trainset, validset, validlabels):
  # instantiate our neuralnet instance
  input_dim = 2240
  hidden_dim = 100
  output_dim = 35

  nn = ReLUbasedFFNeuralNet(input_dim, hidden_dim, hidden_dim, output_dim)

  # our loss function
  lossfn = torch.nn.CrossEntropyLoss(reduction="none")

  # our optimizer and hyperparameter
  learning_rate = 0.01
  target_momentum = 0.8
  optimizer = torch.optim.SGD(nn.parameters(), lr=learning_rate, momentum=target_momentum, weight_decay=0.01)

  # epochs
  epochs = 2

  train_loop(nn, lossfn, optimizer, epochs, trainset, validset, validlabels)
  return nn

# test our accuracy
def test(nn, testset, testlabel, charset_dict):
  correct = 0
  total = 0
  for i, word in enumerate(testset):
    #if i > 50:
    #  break
    #predicted_word = ""
    for j, char in enumerate(word):
      if ord(char) != 183:
        continue
      predicted_ve = nn(compute_ve(char))
      predicted_char = torch.argmax(o_ve)
      predicted_word += chr(predicted_char)

      if predicted_char == charset[valid_label[k][j]]:
        correct += 1
      total += 1
  
  accuracy = 100 * correct / total
  print("accuracy = ", accuracy)

# extractTokens
def extractTokens(texts):
    tokens = []
    texts = re.sub(r'[\n]', ' ', texts)
    words = texts.split(' ')
    stop_words = ['unk', '·nk', 'u·k', 'un·', '·unk', 'unk·', '', '<unk>']
    tokens += [x.lower() for x in words if x not in stop_words]
    return tokens

# read data
def read_data(filepath, filterCorruption='Clean', filterStopWords=True):
    with open(filepath) as f:
        texts = f.read().lower()

    if filterStopWords == True:
        words = extractTokens(texts)

    if filterCorruption == 'All':
        return words

    elif filterCorruption == 'Corrupt':
        corrupted_words = []
        for word in words:
            if chr(183) in word:
                corrupted_words.append(word)
        return corrupted_words
    else:
        clean_words = []
        for word in words:
            if not chr(183) in word:
                clean_words.append(word)
        return clean_words
    
# our main :)
trainset = read_data("/Users/clarissacheam/Documents/Northwestern/Machine Learning/Final Project/wiki.train.dot", 'Clean')
validset = read_data("/Users/clarissacheam/Documents/Northwestern/Machine Learning/Final Project/wiki.valid.dot", 'All')
validlbl = read_data("/Users/clarissacheam/Documents/Northwestern/Machine Learning/Final Project/wiki.valid.txt", 'All')
testset = read_data("/Users/clarissacheam/Documents/Northwestern/Machine Learning/Final Project/wiki.test.dot", 'All')
testlbl = read_data("/Users/clarissacheam/Documents/Northwestern/Machine Learning/Final Project/wiki.test.txt", 'All')

print("trainset size = ", len(trainset))
print("validset size = ", len(validset))
print("validlbl size = ", len(validlbl))
print("testset size = ", len(testset))
print("testlbl size = ", len(testlbl))

print("Start training ...")
nn = train(trainset, validset, validlbl)
print("Training completed")
#test(nn, testset, testlbl, charset_tbl)
print("Test completed")
