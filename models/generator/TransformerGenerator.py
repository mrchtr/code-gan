# --------- source code taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html ---------
# Original Paper and repository here : https://github.com/openai/gpt-2

import math
from abc import ABC

import torch
import torch.nn.functional as f
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer, CrossEntropyLoss
import numpy as np
from transformers import GPTNeoModel, GPT2Model, AutoModelWithLMHead, TransfoXLConfig, GPT2Config
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

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


class PretrainedGPTGenerator(Generator, GenerationMixin, ABC):
    """
    Generator based on the pretrained GPT-Neo of Huggingface
    """

    def __init__(self, config, bos_token, eos_token_id=50256,
                 pad_token_id=50256):
        super(PretrainedGPTGenerator, self).__init__(config)
        self.forward_gumbel = True
        self._config = config
        self.ntoken = config.vocab_size
        configuration = GPT2Config(
            vocab_size=config.vocab_size,
            bos_token_id=bos_token,
            eos_token_id=bos_token,
            pad_token_id=bos_token
        )
        if config.saved_model == "GPT-Code":
            self.transformer = AutoModelWithLMHead.from_pretrained("microsoft/CodeGPT-small-py")
        else:
            self.transformer = GPT2Model.from_pretrained("gpt2-medium", bos_token_id=config.eos_token_id,
                                                     eos_token_id=config.eos_token_id, pad_token_id=config.pad_token_id)
        # self.transformer = GPT2Model.from_pretrained(pretrained_model, bos_token_id=bos_token, eos_token_id=bos_token)
        self.config = self.transformer.config
        self.config.eos_token_id = self._config.eos_token_id
        self.transformer.resize_token_embeddings(self.ntoken)
        self.lm_head = nn.Linear(self.transformer.config.n_embd, self.ntoken)

        # freeze transformer for training
        if config.freezing_generator is True:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def init_state(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def step_forward_gumbel(self, input_ids, gumbel_forward=True, return_dict=None):
        self.forward_gumbel = gumbel_forward
        return self.forward(input_ids, return_dict=return_dict)

    def forward(self, input_ids, hidden=None, prev_hidden=None, return_dict=None, output_attentions=None,
                output_hidden_states=None,
                labels=None, gumbel=True):
        """
        TransformerGAN step forward:
        :param input_ids:
        :param prev_hidden:
        :param return_dict:
        :param output_attentions:
        :param output_hidden_states:
        :param labels:
        :param gumbel:
        :return:
            - pred: used for adversarial training backwards step, and sample generation
        """
        transformer_outputs = self.transformer(input_ids)  # encoder
        hidden_states = transformer_outputs[0]
        if self._config.saved_model == 'GPT-Code':
            lm_logits = transformer_outputs.logits
        else:
            lm_logits = self.lm_head(hidden_states)  # linear layer


        if self.forward_gumbel:
            lm_logits = self.add_gumbel(lm_logits, self.device)  # gumbel_t layer

        # needed for sequence generation, not part of the autograde graph
        # softmax operation will be applied inside the huggingface generation module
        if return_dict:
            if self.forward_gumbel:
                lm_logits = torch.mul(lm_logits, self.temperature)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return CausalLMOutputWithCrossAttentions(
                logits=lm_logits,
                loss=loss
            )
        else:
            prediction = f.softmax(lm_logits * self.temperature, dim=1)  # prediction
            next_token = torch.argmax(lm_logits, dim=1).detach()  # next token - not part of autograde graph
            return prediction, None, next_token

    def gen_sample(self, context, sequence_length, batch_size, min_len=0, forward_gumbel=True, is_eval=False):
        if self._config.open_end_generation and is_eval == False:
            min_len = self._config.sequence_length

        if is_eval:
            eos_token = self.config.eos_token_id
        else:
            eos_token = self.config.pad_token_id

        self.forward_gumbel = forward_gumbel

        if self._config.sampling == 'top_k':
            top_k = self._config.top_k
            sample = self.generate(context,
                                   max_length=sequence_length,
                                   min_length=min_len,
                                   eos_token_id=eos_token,
                                   top_k=top_k,
                                   do_sample=True,
                                   temperature=0.95)
        else:
            sample = self.generate(context,
                                   max_length=sequence_length,
                                   min_length=min_len,
                                   num_samples=1,
                                   eos_token_id=eos_token,
                                   num_beams=5,
                                   no_repeat_ngram_size=2)

        if is_eval:
            # pad all seqs to desired length
            out_tensor = sample[0].data.new(*(batch_size, sequence_length)).fill_(self.config.pad_token_id)
            for i, tensor in enumerate(sample):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            out_tensor = out_tensor.to(self._config.device)
            return out_tensor

        return sample

    def predict(self, context, sequence_length, batch_size, min_len=0):
        eos_token = self.config.eos_token_id
        self.forward_gumbel = False
        top_k = self._config.top_k
        return self.generate(context,
                               max_length=sequence_length,
                               min_length=min_len,
                               eos_token_id=eos_token,
                               top_k=top_k,
                               do_sample=True,
                               num_return_sequences=1,
                               temperature=0.95)



