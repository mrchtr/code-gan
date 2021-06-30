from random import randint

from torch.utils.data import Dataset
import torch
import os
from tqdm import tqdm

from utils.Preprocessor import preprocess


class CodeDataset(Dataset):
    """
    Holding the dataset for the training. Basically all source code files located
    under the given path will be loaded and preprocessed.
    """

    def __init__(self, root_dir, tokenizer, block_size, extensions=None):
        """
        :param root_dir: path to the dir where the source code is holded
        :param tokenizer: tokenizer for converting text into vector tokens
        :param block_size: size of input sequence
        :param extensions: files types that should be included
        """

        assert os.path.isdir(root_dir), f"Input file path {root_dir} not found"
        self.root_dir = root_dir
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.extensions = extensions

        # all relevant files of the given root dir
        self.files = self.load_files()
        text = ""
        for f in tqdm(self.files, unit="files", position=0, leave=True):
            with open(f, 'r', encoding="utf-8") as file:
                content = self.preprocess_file(file.read())
                text = text + content

        # convert extracted text to tokenized trainings data
        self.examples = self.init_examples(text)

        print(f"Dataset initialization done. Loaded content of {len(self.files)} files.")

    def load_files(self):
        return [os.path.join(d, filename)
                for d, dirs, files in os.walk(self.root_dir)
                for filename in files if self.is_relevant_file(filename)]

    def is_relevant_file(self, filename):
        if self.extensions == None:
            return True
        elif type(self.extensions) is list:
            return any(filename.endswith(extension) for extension in self.extensions)
        elif type(self.extensions) is str:
            return filename.endswith(self.extensions)

    def preprocess_file(self, content):
        """
        TODO include a better preprocessing here
        e.g. remove comments, reformat code to common format, remove line breaks and not needed tab spaces etc. ...
        :param content: content of file
        :return: preprocessed file
        """
        content = content.strip()
        #content = preprocess(content)
        return content

    def init_examples(self, text):
        examples = []
        tokenized_text = []
        n = 5000  # just processing 5000 files
        for i in tqdm(range(0, len(text), n), unit="tokens", position=0, leave=True):
            tokenized_text[-1:-1] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text[i:i + n]))
            # all list elements to list
        for i in tqdm(range(0, len(tokenized_text) - self.block_size + 1, self.block_size), unit="samples", position=0,
                      leave=True):
            examples.append(
                self.tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + self.block_size])
            )

        return examples

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __len__(self):
        return len(self.examples) - self.block_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.examples[index]),
            torch.tensor(self.examples[index+1]),
        )

    def get_random_real_sample(self, batch_size):
        samples = [self.__get_random_sample() for _ in range(batch_size)]
        return torch.tensor(samples)

    def __get_random_sample(self):
        rand = randint(0, self.__len__())
        return self.examples[rand]

