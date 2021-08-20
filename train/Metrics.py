from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import jellyfish

class Metrics:
    def __init__(self, max_len):
        # init bert model
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model.eval()

        self.smoothie = SmoothingFunction().method4

        self.max_len = max_len

    def get_embed(self, inp):
        """
        input should be batch of prediciton and refrence
        """
        inputs = self.tokenizer(inp, padding='max_length', max_length=self.max_len, return_tensors="pt", truncation=True)
        context_embed = self.model(**inputs)
        return context_embed

    def get_similarity(self, pred, ref):
        inp = [pred, ref]
        context_embed = self.get_embed(inp)
        out = context_embed[0]
        x = out[0]
        y = out[1]

        euclidian = torch.norm(y - x) / self.max_len
        cosinus_sim = torch.cosine_similarity(out[0].unsqueeze(0), out[1].unsqueeze(0))
        cosinus_sim = torch.mean(cosinus_sim)

        levenstein_dis = jellyfish.levenshtein_distance(pred, ref)

        pred_tokenized = self.tokenizer.encode(pred)
        ref_tokenized = self.tokenizer.encode(ref)
        bleu = sentence_bleu([ref_tokenized], pred_tokenized, smoothing_function=self.smoothie)

        return euclidian.item(), cosinus_sim.item(), levenstein_dis, bleu

