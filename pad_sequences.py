''' This contains the packing and unpacking the padded sequences for rnn.

This contains code from https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

we want to run a LSTM on a batch of 3 character sequences ['long_str', 'tiny', 'medium']

step 1 : construct vocabulary
step 2 : convert the sequences into numerical form
step 3 : define model
step 4 : prepare data, by padding with 0 (<pad> token), making the batch equal lengths
step 5 : sort the data in the batch in descending order by their original lengths
step 6 : apply the embedding layer for the batch
step 7 : call the pack_padded_sequences fn, with embeddings and lengths of original sentences
step 8 : call the forward method of the lstm model
step 9 : call the unpack_padded_sequences (pad_packed_sequence) method if required
'''

import torch
import torch.nn as nn
import torch.optim as optim

from torch import LongTensor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

data = ['long_str', 'tiny', 'medium']

# step 1 : construct vocabulary
vocab = ['<pad>'] + sorted(set([char for seq in data for char in seq]))
# vocab = ['<pad>', '_', 'd', 'e', 'g', 'i', 'l', 'm', 'n', 'o', 'r', 's', 't', 'u', 'y']

# step 2 : convert the sequences into numerical form
vectorized_data = [[vocab.index(tok) for tok in seq] for seq in data]
# vectorized_data = [[6, 9, 8, 4, 1, 11, 12, 10], [12, 5, 8, 14], [7, 3, 2, 5, 13, 7]]

# step 3 : define model

# input for embedding layer is lengths of inputs
# output for embedding layer is embedding shape of inputs
embedding_layer = nn.Embedding(len(vocab), 4)

# input_size is the embedding output size
# hidden_size is the hidden size of lstm
lstm = nn.LSTM(input_size=4, hidden_size=5, batch_first=True)

# step 4 : prepare data, by padding with 0 (<pad> token), making the batch equal lengths
