from os.path import exists

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer
import os.path
import sentencepiece as spm
from utils.FileUtils import get_all_files, create_dir_if_not_exists


class SentencepieceResolver:
    """
    Wrapper for a basic byte pair tokenizer based on sentencepiece.
    """

    def __init__(self, config, path=".", special_tokens=[], vocab_size=5000, name="m"):
        """
        :param path: path to the training files
        :param special_tokens: special tokens
        """
        self.path = config.training_data
        self.special_tokens = config.special_tokens
        self.vocab_size = config.vocab_size
        self.name = name

        self.tokenizer = None

        self.train()
        self.load()

    def train(self):
        if exists(f"{self.name}.model"):
            print("Skip training. Model with the name already exists.")
            return
        self.tokenizer = spm.SentencePieceTrainer.train(input=self.path, model_prefix=self.name, vocab_size=self.vocab_size, user_defined_symbols=self.special_tokens)

    def load(self):
        sp = spm.SentencePieceProcessor()
        model = f"{self.name}.model"
        sp.Load(model)
        self.tokenizer = sp

    def encode(self, inp):
        if self.tokenizer is None:
            return
        return self.tokenizer.encode(inp)

    def decode(self, inp):
        if self.tokenizer is None:
            return
        return self.tokenizer.decode(inp)

class CodeTokenizerResolver:
    """
    CodeTokenizer is a load a byte-lever-tokenizer that is trained on source code.
    If the checkpoint files not exist, the training for the source code tokenizer is started.
    """
    def __init__(self, path=".", config=None, min_frequency=2, pretrained="code-tokenizer", training_files=None) \
            -> GPT2Tokenizer:
        """
        :param path: path to pretrained tokenizer checkpoints files
        :param vocab_size: vocabulary size
        :param min_frequency:
        :param pretrained: name for pretrained files
        :param training_files: files for the training process
        """
        self.config = config
        self.vocab_size = config.vocab_size
        self.path = path
        self.min_frequency = min_frequency
        self.tokenizer = self.train_if_not_available(path, pretrained, config.data_dir, self.vocab_size, min_frequency)

    def train_if_not_available(self, path, tokenizer_name, training_files, vocab_size, min_frequency):
        tokenizer_wt = path + "/" + tokenizer_name + "-vocab.json"
        tokenizer_m = path + "/" + tokenizer_name + "-merges.txt"
        if not os.path.isfile(tokenizer_wt) or not os.path.isfile(tokenizer_m):
            print("INFO: Start training the code tokenizer ... ")
            create_dir_if_not_exists(path)
            training_files = get_all_files(training_files)

            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(files=training_files, vocab_size=vocab_size, min_frequency=min_frequency,
                            special_tokens=self.config.special_tokens)

            tokenizer.save_model(path, tokenizer_name)

        return self.load_pretrained(path, tokenizer_name)

    @staticmethod
    def load_pretrained(path, name):
        """
        init tokenizer by checkpoint files
        :param path: path to checkpoints
        :param name: name of file
        :return:
        """
        return GPT2Tokenizer(
            path + "/" + name + "-vocab.json",
            path + "/" + name + "-merges.txt"
        )

    def encode(self, inp):
        if self.tokenizer is None:
            return
        return self.tokenizer.encode(inp)

    def decode(self, inp):
        if self.tokenizer is None:
            return
        return self.tokenizer.decode(inp)
