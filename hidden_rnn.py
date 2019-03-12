'''
This code explains the hidden states and outputs of rnn.
This is code reference for the blog post: https://graviraja.github.io/unwraprnn/
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import LongTensor

# create a dummy data of batch_size = 3
data = ['long_str', 'tiny', 'medium']

# create the vocabulary
vocab = ['<pad>'] + sorted(set([char for seq in data for char in seq]))
# vocab = ['<pad>', '_', 'd', 'e', 'g', 'i', 'l', 'm', 'n', 'o', 'r', 's', 't', 'u', 'y']

# convert into numerical form
vectorized_data = [[vocab.index(tok) for tok in seq] for seq in data]
# vectorized_data = [[6, 9, 8, 4, 1, 11, 12, 10], [12, 5, 8, 14], [7, 3, 2, 5, 13, 7]]

# prepare data, by padding with 0 (<pad> token), making the batch equal lengths
seq_lengths = LongTensor([len(seq) for seq in vectorized_data])
sequence_tensor = Variable(torch.zeros(len(vectorized_data), seq_lengths.max(), dtype=torch.long))

for idx, (seq, seq_len) in enumerate(zip(vectorized_data, seq_lengths)):
    sequence_tensor[idx, :seq_len] = LongTensor(seq)

# sequence_tensor = ([[ 6,  9,  8,  4,  1, 11, 12, 10],
#                     [12,  5,  8, 14,  0,  0,  0,  0],
#                     [ 7,  3,  2,  5, 13,  7,  0,  0]])

# convert the input into time major format
sequence_tensor = sequence_tensor.t()
# sequence_tensor shape => [max_len, batch_size]

input_dim = len(vocab)
print(f"Length of vocab : {input_dim}")

# hidden dimension in the RNN
hidden_dim = 5

# embedding dimension
embedding_dim = 5


class Single_Layer_Uni_Directional_RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)

    def forward(self, input):
        # input shape => [max_len, batch_size]

        embed = self.embedding(input)
        # embed shape => [max_len, batch_size, embedding_dim]

        output, hidden = self.rnn(embed)
        # output shape => [max_len, batch_size, hidden_size]
        # hidden shape => [1, batch_size, hidden_size]

        return output, hidden

n_layers = 1
bidirectional = False
model = Single_Layer_Uni_Directional_RNN(input_dim, embedding_dim, hidden_dim, n_layers, bidirectional)
output, hidden = model(sequence_tensor)

print(f"Input shape is : {sequence_tensor.shape}")
print(f"Output shape is : {output.shape}")
print(f"Hidden shape is : {hidden.shape}")

assert (output[-1, :, :] == hidden[0]).all(), "Final output must be same as Hidden state in case of Single layer uni-directional RNN"


class Multi_Layer_Uni_Directional_RNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class Single_Layer_Bi_Directional_RNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class Multi_Layer_Bi_Directional_RNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
