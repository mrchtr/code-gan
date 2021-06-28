from torch import nn


class Discriminator(nn.Module):
    """
    Simple discriminator: takes a vector of tokens and mapped these to one final output state
    """
    def __init__(self, embedding_dim: int):
        super(Discriminator, self).__init__()
        self.embedding_size = embedding_dim
        self.dense = nn.Linear(int(embedding_dim), 1)


    def forward(self, input):
        return self.dense(input)
