from os.path import exists

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer
import os.path

from utils.FileUtils import get_all_files, create_dir_if_not_exists

class CodeTokenizerResolver:
    """
    CodeTokenizer is a byte-lever-tokenizer that is trained on source code.
    If the checkpoint files not exist, the training for the source code tokenizer is started.
    Otherwise the tokenizer will be initialized with the existing checkpoints.
    """

    def __init__(self, path=".", config=None, min_frequency=2, pretrained="code-tokenizer", training_files=None) \
            -> ByteLevelBPETokenizer:
        """
        :param path: path to pretrained tokenizer checkpoints files
        :param vocab_size: vocabulary size
        :param min_frequency:
        :param pretrained: name for pretrained files
        :param training_files: files for the training process
        """
        self.config = config
        self.path = path
        self.min_frequency = min_frequency
        self.tokenizer = self.train_if_not_available(path, pretrained, config.validation_data, min_frequency)

    def get(self):
        return self.tokenizer

    def train_if_not_available(self, path, tokenizer_name, training_files, min_frequency):
        tokenizer_wt = path + "/" + tokenizer_name + "-vocab.json"
        tokenizer_m = path + "/" + tokenizer_name + "-merges.txt"
        if not os.path.isfile(tokenizer_wt) or not os.path.isfile(tokenizer_m):
            print("INFO: Start code tokenizer training ... ")
            create_dir_if_not_exists(path)
            training_files = get_all_files(training_files)

            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(files=training_files, vocab_size=self.config.vocab_size, min_frequency=min_frequency,
                            show_progress=True,
                            special_tokens=self.config.special_tokens)

            tokenizer.save_model(path, tokenizer_name)
        else:
            print("Using pretrained tokenizer ...")

        return self.load_pretrained(path, tokenizer_name)

    def load_pretrained(self, path, name):
        """
        init tokenizer by checkpoint files
        :param path: path to checkpoints
        :param name: name of file
        :return:
        """
        tokenizer = ByteLevelBPETokenizer(
            path + "/" + name + "-vocab.json",
            path + "/" + name + "-merges.txt",

        )
        tokenizer.add_special_tokens(self.config.special_tokens)
        return tokenizer

    def encode(self, inp):
        if self.tokenizer is None:
            return
        return self.tokenizer.encode(inp)

    def decode(self, inp):
        if self.tokenizer is None:
            return
        return self.tokenizer.decode(inp)
