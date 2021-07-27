from random import randint

from torch import nn
import torch.nn.functional as f
import torch
from models.generator.RelationalMemory import RelationalMemory


class Generator(nn.Module):
    """
    Boilerplate class for different generator models.
    """
    def __init__(self, config):
        super(Generator, self).__init__()
        self.temperature = config.temperature

        self.device = config.device

    def sample(self, context, sequence_length):
        """
        Sample method that generates a fake sequence based on the given context
        :param context: context for sequence generation
        :return:
        """
        # return torch.randint(0, 150, (sequence_length, ), dtype=torch.float32)  # todo implement real sampling
        NotImplementedError()

    @staticmethod
    def add_gumbel(o_t, device, eps=1e-10):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size())
        u = u.to(device)

        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t

class GeneratorRelGAN(Generator):
    """
    Text generator based on a Relational Memory like proposed by Nie et al. 2019
    """
    def __init__(self, n_vocab, embedding_dim, mem_slots=1, head_size=256, num_heads=2, drop_prob=0.2):
        super(GeneratorRelGAN, self).__init__()
        self.embeddings = nn.Embedding(n_vocab, embedding_dim)
        self.hidden_dim = mem_slots * num_heads * head_size
        self.rm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                     num_heads=num_heads, return_all_outputs=True)
        self.rm2out = nn.Linear(self.hidden_dim, n_vocab)

    def init_state(self, batch_size):
        memory = self.rm.initial_state(batch_size)
        memory = self.rm.repackage_hidden(memory)  # detach memory at first
        return memory

    def forward(self, x, hidden):
        """
        RelGAN step forward
        :param x: input context / noise
        :param hidden: memory size - previous state
        :return: pred, hidden, next_token
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: next sentence token
        """
        emb = self.embeddings(x).squeeze(1)
        out, hidden = self.rm(emb, hidden)
        gumbel_t = self.add_gumbel(self.rm2out(out.squeeze(1)), self.device)
        next_token = torch.argmax(gumbel_t, dim=1).detach()

        pred = f.log_softmax(gumbel_t * self.temperature, dim=-1)

        return pred, hidden, next_token

class GeneratorLSTM(Generator):
    """
    Basic implementation of a LSTM that generates code suggestion based on given context
    """

    def __init__(self, config):
        """
        :param n_vocab: Size of vocabulary
        :param embedding_dim: Dimension of the embeddings in the lookup table
        :param hidden_dim: Neurons in a hidden layer
        :param num_layers: Number of layers
        :param sequence_length: sequence length of output
        :param drop_prob: dropout probability
        """
        super(GeneratorLSTM, self).__init__(config)
        self.config = config
        self.n_vocab = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.nhid
        self.num_layers = config.nlayers

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)  # input: batch_size * seq_len / output: batch_size * seq_len * embed_dim
        output, state = self.lstm(embed, prev_state)  # seq_length * batch_size * input_size

        #  output.shape:  batch_size * sequence_length * n_vocab
        gumbel_t = self.add_gumbel(self.fc(output.squeeze(1)), self.device)  # gumbel softmax trick proposed by Nie et al. 2019
        next_token = torch.argmax(gumbel_t, dim=1).detach()  # batch_size * 1
        prediction = f.log_softmax(gumbel_t * self.temperature, dim=-1) # batch_size * n_vocab
        return prediction, state, next_token

    def sample(self, context, sequence_length, batch_size, num_samples=1):
        """
        Generating sample of context
        TODO: for now just the prediction of next token - apply different encoding strategies later
        :param context: previous token
        :param sequence_length: length of sample
        :return:
        """
        global all_preds # batch_size * seq_len * vocab_size
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, sequence_length).long()
        samples.to(self.device)

        for b in range(num_batch):
            hidden = self.init_state(batch_size)
            inp = context
            for i in range(sequence_length):
                pred, hidden, next_token = self.forward(inp, hidden)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                inp = next_token.reshape(batch_size, 1)
        samples = samples[:num_samples]  # num_samples * seq_len

        return samples

    def random_start_letter(self):
        return randint(1, self.n_vocab-1)

    def init_state(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        return h, c
