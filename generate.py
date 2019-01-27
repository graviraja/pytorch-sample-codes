""" Char level RNN to generate words.
We try to generate surnames from 18 languages given a language code.
"""

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS character


def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s):
    # convert the unicode string to plain ascii
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

category_lines = {}
all_categories = []


def readLines(filename):
    # read a file and split into lines.
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


def letterToIndex(letter):
    # find the index of given letter from all the letters.
    return all_letters.find(letter)


def letterToTensor(letter):
    # convert the letter into a tensor of shape (1, n_letters)
    # shape is (1, n_letters) instead of (n_letters) because 1 is batch_size
    # pytorch expects everything in batches
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    # convert the line into a tensor of shape (line_length, 1, n_letters)
    # 1 in shape is batch size
    tensor = torch.zeros(len(line), 1, n_letters)
    for index, letter in enumerate(line):
        tensor[index][0][letterToIndex(letter)] = 1
    return tensor


def randomChoice(l):
    # random pick a value from the list l
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExamples():
    # randomly pick a category and randomly pick a line from that category
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


def categoryToTensor(category):
    # convert the category to tensor of shape (1, n_catergories)
    tensor = torch.zeros(1, n_categories)
    tensor[0][all_categories.index(category)] = 1
    return tensor


def targetTensor(line):
    # create the target tensor from the line, shifted by one letter
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        combined_output = torch.cat((hidden, output), 1)
        output = self.o2o(combined_output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNN(n_letters, 128, n_letters)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=1e-4)
learning_rate = 0.0005


all_losses = []
total_epochs = 100000
avg_loss = 0

for epoch in range(total_epochs):
    # forward pass
    category, line = randomTrainingExamples()
    category_tensor = categoryToTensor(category)
    line_tensor = lineToTensor(line)
    target_tensor = targetTensor(line)

    # reshape the target tensor suitable for loss
    target_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    loss = 0
    for char in range(len(line)):
        output, hidden = rnn(category_tensor, line_tensor[char], hidden)
        l = criterion(output, target_tensor[char])
        loss += l

    # backward pass
    loss.backward()
    optimizer.step()
    avg_loss += loss / len(line)
    if (epoch + 1) % 5000 == 0:
        print(f"epoch : {epoch}, loss : {avg_loss.item() / epoch}")
        avg_loss = 0

torch.save(rnn, 'char-rnn-generation.pt')
rnn = torch.load('char-rnn-generation.pt')

max_length = 20


def sample(category, start_letter):
    # sample a word using the category code and starting letter
    with torch.no_grad():
        category_tensor = categoryToTensor(category)
        line_tensor = lineToTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, line_tensor[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
                input_tensor = lineToTensor(letter)
        return output_name


def samples(category, start_letters='ABC'):
    # create samples from a category and list of starting words
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')
