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
                 vocab_size,
                 embedding_dim,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
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
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0,
                                      max_norm=5.0)

        # Convolutional Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        # Fully-connected layer
        self.fc = nn.Linear(np.sum(num_filters), num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inp):
        """
        Perform a forward pass through the network.
        :param inp: A tensor of token ids with shape (batch_size, sequence_length)
        :return: logits (torch.Tensor): Output logits with shape (batch_size, n_classes)
        """

        # Get embeddings from `inp`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(inp).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [f.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [f.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits