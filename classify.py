""" Char level RNN to classify words.
We try to classify surnames from 18 languages.
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

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


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


def categoryFromOutput(output):
    # return the category and category id from the output tensor
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    # random pick a value from the list l
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExamples():
    # randomly pick a category and randomly pick a line from that category
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, 128, n_categories)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.005)


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    optimizer.zero_grad()
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()


def evaluate(line_tensor):
    with torch.no_grad():
        hidden = rnn.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        return output

avg_loss = 0
all_losses = []
for epoch in range(100000):
    category, line, category_tensor, line_tensor = randomTrainingExamples()
    output, loss = train(category_tensor, line_tensor)
    avg_loss += loss

    if (epoch + 1) % 5000 == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print(f"epoch : {epoch}, loss : {avg_loss / epoch}, {line} / {guess} {correct} ({category})")
        all_losses.append(avg_loss)
        avg_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')
rnn = torch.load('char-rnn-classification.pt')

confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000
# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExamples()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
