# --------- source code taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html ---------
#Original Paper and repository here : https://github.com/openai/gpt-2

import math
import torch
import torch.nn.functional as f
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from transformers import GPTNeoModel

from models.generator.Generator import Generator


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerGenerator(Generator):
    """
    Model followed concept of 'Attention is all you need'.
    Default parameters are same that are used in gpt-2
    """
    def __init__(self, config):
        super(TransformerGenerator, self).__init__(config)
        self.config = config
        self.pos_encoder = PositionalEncoding(config.ninp, config.dropout)  # positional encoder
        encoder_layers = TransformerEncoderLayer(config.ninp, config.nhead, config.nhid, config.dropout)  # encoder stack
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.nlayers)
        self.encoder = nn.Embedding(config.vocab_size, config.ninp)
        self.ninp = config.ninp
        self.decoder = nn.Linear(config.ninp, config.vocab_size)

        self.init_weights()

    def init_state(self, sz):
        """
        Generates square subsequent mask. For consitency method is called init state.
        :param sz: batch_size
        :return:
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(self.config.device)
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        gumbel_t = self.add_gumbel(output.squeeze(1), self.device)
        next_token = torch.argmax(gumbel_t, dim=2).detach()[:, -1]  # batch_size * 1
        prediction = f.log_softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * n_vocab
        prediction = prediction[:, -1, :]  # just returning the next token, cut of the first of each batch
        return prediction, src_mask, next_token

    def sample(self, context, sequence_length, batch_size, num_samples=1):
        """
                Generating sample of context
                TODO: for now just the prediction of next token - apply different encoding strategies later
                :param context: previous token
                :param sequence_length: length of sample
                :return:
                """
        global all_preds  # batch_size * seq_len * vocab_size
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, sequence_length).long()
        samples.to(self.device)
        src_mask = self.init_state(batch_size)
        for b in range(num_batch):
            inp = context
            for i in range(sequence_length):
                print(f"device inp: {inp.device} device src_mask: {src_mask.device}")
                pred, src_mask, next_token = self.forward(inp, src_mask)
                print(f"device next_token: {inp.device} device src_mask: {src_mask.device}")
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                inp = torch.from_numpy(np.append(inp.cpu(), next_token.unsqueeze(1).cpu(), axis=1)[:,1:]).to(self.device)
        samples = samples[:num_samples]  # num_samples * seq_len

        return samples


class PretrainedGPTGenerator(Generator):
    """
    Generator based on the pretrained GPT-Neo of Huggingface
    """

    def __init__(self, ntoken, pretrained_model="EleutherAI/gpt-neo-125M"):
        super(PretrainedGPTGenerator, self).__init__()
        self.ntoken = ntoken
        self.transformer = GPTNeoModel.from_pretrained(pretrained_model)
        self.decoder = nn.Linear(self.transformer.config.hidden_size, ntoken)


    def init_state(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        output = self.transformer(input_ids=src, attention_mask=src_mask)



