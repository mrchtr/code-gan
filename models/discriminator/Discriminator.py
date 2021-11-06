from torch import nn
import torch.nn.functional as f
import torch
import numpy as np
import math

from transformers import AutoModel


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

class CodeBertDiscriminator(Discriminator):

    def __init__(self, base_model, config):
        super(Discriminator, self).__init__()
        self.encoder = base_model
        self.dropout = nn.Dropout(0.5)
        # linear layer
        self.dense = nn.Linear(self.encoder.config.hidden_size, 1)

        # freezing model
        for param in self.encoder.parameters():
            param.requieres_grad = False
    def embed(self, inp):
        return self.encoder.roberta(inp).last_hidden_state

    def forward(self, inp = None, embedding = None):
        if inp is not None:
            encoded = self.encoder.roberta(inp)
        elif embedding is not None:
            encoded = self.encoder.roberta(inputs_embeds=embedding)
        else:
            raise("At least on of both parameter have to be not empty.")

        pred = encoded[0]
        return self.dense(pred)

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
        self.embeddings = nn.Linear(config.vocab_size, config.discriminator_embedding_dim, bias=False)
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

class RelGAN_D(CNNDiscriminator):
    def __init__(self,config, num_filters=[300, 300, 300, 300], filter_sizes=[2, 3, 4, 5], num_rep=1, dropout=0.25):
        super(RelGAN_D, self).__init__(
            config,
            filter_sizes=[2, 3, 4, 5],
            num_filters=[300, 300, 300, 300],
            num_classes=2,
            dropout=0.5
        )

        self.embed_dim = config.discriminator_embedding_dim
        self.max_seq_len = (config.sequence_length)
        self.feature_dim = sum(num_filters)
        self.emb_dim_single = int(self.embed_dim / num_rep)

        self.embeddings = nn.Linear(self.vocab_size, self.embed_dim, bias=False)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1, n, (x, self.emb_dim_single), stride=(1, self.emb_dim_single)
                )
                for (n, x) in zip(num_filters, filter_sizes)
            ]
        )

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp = None, embedding = None):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(
            1
        )  # batch_size * 1 * max_seq_len * embed_dim

        cons = [
            f.relu(conv(emb)) for conv in self.convs
        ]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [
            f.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons
        ]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = (
            pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)
        )  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = (
                torch.sigmoid(highway) * f.relu(highway)
                + (1.0 - torch.sigmoid(highway)) * pred
        )  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                torch.nn.init.uniform_(param, a=-0.05, b=0.05)
