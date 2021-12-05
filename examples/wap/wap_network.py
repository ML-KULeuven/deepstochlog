from pathlib import Path
from typing import List

import torch
import torch.nn as nn

embed_size = 256


vocab_location = (
    Path(__file__).parent / ".." / ".." / "data" / "raw" / "wap" / "vocab_746.txt"
)
vocab = dict()
with open(vocab_location) as f:
    for i, word in enumerate(f):
        word = word.strip()
        vocab[word] = i


def tokenize(sentence):
    sentence = sentence.split(" ")
    tokens = []
    numbers = list()
    indices = list()
    for i, word in enumerate(sentence):
        if word.isdigit():
            numbers.append(int(word))
            tokens.append("<NR>")
            indices.append(i)
        else:
            if word in vocab:
                tokens.append(word)
            else:
                tokens.append("<UNK>")
    return [vocab[token] for token in tokens], numbers, indices


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, device=None, p_drop=0.0):
        super(RNN, self).__init__()
        self.lstm = nn.GRU(
            embed_size, hidden_size, 1, bidirectional=True, batch_first=True
        )
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            vocab_size, embed_size
        )  # , _weight=torch.nn.Parameter(weights))
        self.dropout = nn.Dropout(p_drop)
        self.device = device

    def forward(self, sentence_input: List[List[str]]):

        tokenizations = [tokenize(s[0].strip('"').strip()) for s in sentence_input]

        # TODO: Implement this as batch instead
        tensors = []
        for tokenization in tokenizations:
            x, _, indices = tokenization
            n1, n2, n3 = indices
            seq_len = len(x)
            x = torch.LongTensor(x).unsqueeze(0).to(self.device)
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = x.view(seq_len, 2, self.hidden_size)
            x1 = torch.cat([x[-1, 0, ...], x[n1, 0, ...], x[n2, 0, ...], x[n3, 0, ...]])
            x2 = torch.cat([x[0, 1, ...], x[n1, 1, ...], x[n2, 1, ...], x[n3, 1, ...]])
            x = torch.cat([x1, x2])
            #        return x
            tensors.append(self.dropout(x))

        result = torch.stack(tensors)
        if self.device:
            result = result.to(self.device)

        return result
