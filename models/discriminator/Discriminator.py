from torch import nn
import torch.nn.functional as f
import torch
import numpy as np
import math


class Discriminator(nn.Module):
    """
    Simple discriminator: takes a vector of tokens and mapped these to one final output state
    """
    def __init__(self, embedding_size: int):
        super(Discriminator, self).__init__()
        self.embedding_size = embedding_size
        self.dense = nn.Linear(int(embedding_size), 1)

    def forward(self, inp):
        return self.dense(inp.type(torch.float32))


class CNNDiscriminator(Discriminator):
    """
    https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
    """

    def __init__(self,
                 config,
                 filter_sizes=[2, 3, 4, 5],
                 num_filters=[300, 300, 300, 300],
                 num_classes=2,
                 dropout=0.5):

        """
        Constructor of the basic cnn classifier.
        :param vocab_size: Size of the vocabulary
        :param embedding_dim: Dimensions of word vectors. For BPE: sequence_length
        :param filter_sizes: List of filters
        :param num_filters: List of number of filter, has the same length as 'filter_sizes'
        :param num_classes: Num of classes that should be classified. Default 2: Fake or Real
        :param dropout: Dropout rate. Defaul: 0.5
        """
        super(Discriminator, self).__init__()
        self.device = config.device
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.discriminator_embedding_dim
        self.feature_dim = sum(num_filters)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embedding_dim,
                                      padding_idx=0,
                                      max_norm=5.0)

        # Convolutional Network
        self.conv2d_list = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.embedding_dim), stride=(1, self.embedding_dim)) for (n, f) in
            zip(num_filters, filter_sizes)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        """
        Perform a forward pass through the network.
        :param inp: A tensor of token ids with shape (batch_size, sequence_length)
        :return: logits (torch.Tensor): Output logits with shape (batch_size, n_classes)
        """

        # Get embeddings from `inp`. Output shape: batch_size * 1 * max_seq_len * embed_dim
        x_embed = self.embedding(inp).unsqueeze(1).float()
        cons = [f.relu(conv(x_embed)) for conv in self.conv2d_list]
        pools = [f.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * f.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)

        return logits