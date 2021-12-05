from typing import List

from torch import nn as nn
import torch.nn.functional as F
import torch


class SymbolEncoder(nn.Module):
    def __init__(self):
        super(SymbolEncoder, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )

        self.mlp = nn.Sequential(
            nn.Linear(16 * 11 * 11, 128),
            nn.ReLU(),
            # nn.Dropout2d(0.8)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x


class SymbolClassifier(nn.Module):
    def __init__(self, encoder, N=10):
        super(SymbolClassifier, self).__init__()
        self.encoder = encoder
        self.fc2 = nn.Linear(128, N)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x


class MNISTNet(nn.Module):
    def __init__(self, output_features=10, with_softmax=True):
        super(MNISTNet, self).__init__()
        self.with_softmax = with_softmax
        if with_softmax:
            self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_features),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


# Maybe later change with https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
def get_word_vector_model():
    import spacy

    # Load the spacy model that you have installed
    try:
        nlp = spacy.load("en_core_web_md")
    except IOError:
        # Word2Vec model is not loaded, let's load it!
        import os

        os.system("python -m spacy download en_core_web_md")
        nlp = spacy.load("en_core_web_md")
    return nlp


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class EmbeddingsFFNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, X):
        return self.net(X)

    def predict(self, X):
        Y_pred = self.forward(X)
        y_pred = torch.max(Y_pred, 1)
        predicted_labels = y_pred.indices
        return predicted_labels


# def get_pretrained_model(path: str):
#     model = MNISTNet()
#     model.load_state_dict(torch.load(path))
#     return model
#
#
class Classifier(nn.Module):
    def __init__(
        self, input_size: int, num_outputs: int = 10, with_softmax: bool = True
    ):
        super(Classifier, self).__init__()
        self.with_softmax = with_softmax
        self.input_size = input_size
        if with_softmax:
            self.softmax = nn.Softmax(1)
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, num_outputs),
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.net(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        *sizes,
        encoder=nn.Identity(),
        activation=nn.ReLU,
        softmax=True,
        batch=True
    ):
        super(MLP, self).__init__()
        layers = []
        self.batch = batch
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if softmax:
            layers.append(nn.Softmax(-1))
        self.nn = nn.Sequential(*layers)
        self.encoder = encoder

    def forward(self, x):
        if not self.batch:
            x = x.unsqueeze(0)
        x = self.encoder(x)
        x = self.nn(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, output_features=10, with_softmax=True):
        super(ImageEncoder, self).__init__()
        self.with_softmax = with_softmax
        if with_softmax:
            self.softmax = nn.Softmax(1)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.linear_encoder = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_features),
        )

    def forward(self, x):
        x = self.conv_encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.linear_encoder(x)
        return x


class LSTMSequenceImage(nn.Module):
    """Example usage:

    A = [torch.rand(l, 1, 28, 28) for l in [10,2,7]] # 3 sequences of 10,2,7 elements
    image_encoder_size = 50
    rnn_hidden_size = 100
    image_encoder = ImageEncoder(output_features = image_encoder_size)
    output = LSTMSequenceImage(image_encoder,image_encoder_size,rnn_hidden_size)(A) # tensor with 3 values in [0,1]

    """

    def __init__(
        self, image_encoder, encoder_output_size: int, rnn_hidden_size=10  # MnistModel
    ):
        super().__init__()

        self.encoder_output_size = encoder_output_size
        self.image_encoder = image_encoder
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(
            input_size=self.encoder_output_size, hidden_size=self.rnn_hidden_size
        )
        self.fc = nn.Sequential(nn.Linear(self.rnn_hidden_size, 1), nn.Sigmoid())

    def forward(self, input_sequences: List[torch.Tensor]):
        batch_size = len(input_sequences)

        # Keep track if the indices of the last element of the sequences before pad
        last_indices = [len(i) - 1 for i in input_sequences]

        # Padding the sequence
        neural_input = torch.nn.utils.rnn.pad_sequence(
            input_sequences, batch_first=True
        )

        # Reshaping to a 4-dimensional tensor (as required by MNISTNet()
        first_two = neural_input.shape[:2]
        neural_input = neural_input.view(-1, 1, 28, 28)

        # Encoding images in the sequence
        embedded_sequence = self.image_encoder(neural_input)

        # Restoring batch and sequence dimensions
        embedded_sequence = embedded_sequence.view(*(list(first_two) + [-1]))

        # Processing sequences with RNNs
        outputs, _ = self.rnn(embedded_sequence)

        # Taking the output for the last element of each  sequence
        outputs = outputs[torch.arange(batch_size), last_indices]

        # Project it with a Linear layer with 1 output and sigmoid on top
        y = self.fc(outputs).squeeze(-1)
        return y
