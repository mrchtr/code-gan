from random import randint

from torch.utils.data import Dataset
import torch
import os
from tqdm import tqdm

from utils.Preprocessor import preprocess

class TextDataset(Dataset):
    """
    Dataset for training the gan. Holding tokenized text.
    """

    def __init__(self, block_size, inp, eos_token_id=2, pad_token_id=None):
        """
        :param block_size: size of sequences
        :param inp: input tokens as vector representation
        """

        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.block_size = block_size
        self.data = inp


    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        offset = index + self.block_size

        return torch.tensor(self.data[index:offset]),



    def get_random_context_with_ground_truth(self, batch_size, start_len, seq_len):
        ground_truth_len = seq_len - start_len
        context = []
        ground_truth = []
        for _ in range(batch_size):
            sample = self.__get_random_sample(seq_len)  # full max len sample (context, ground_truth)
            context.append(sample[0:start_len])  # build context

            _ground_truth = sample[start_len:seq_len]
            if self.eos_token_id in _ground_truth:
                # get index of <EOL> token
                index = _ground_truth.index(self.eos_token_id)
                # slicing after + fill of with <EOL> tokens
                _ground_truth = _ground_truth[0:index] + [self.pad_token_id] * (ground_truth_len - index)
            _ground_truth = sample[0:start_len] + _ground_truth
            ground_truth.append(_ground_truth)
        return torch.tensor(context), torch.tensor(ground_truth)

    def get_random_real_sample(self, batch_size, seq_len):
        samples = [self.__get_random_sample(seq_len) for _ in range(batch_size)]
        return torch.tensor(samples)

    def __get_random_sample(self, seq_len):
        max_len = self.__len__() - seq_len - 1
        rand = randint(0, max_len)
        return self.data[rand:rand+seq_len]

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
        content = preprocess(content)
        return content

    def init_examples(self, text):
        tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return self.tokenizer.build_inputs_with_special_tokens(tokenized_text)


    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __len__(self):
        return len(self.examples) - self.block_size

    def __getitem__(self, index):
        offset = index + self.block_size
        return (
            torch.tensor(self.examples[index:offset]),
            torch.tensor(self.examples[index+1:offset+1]),
        )

    def get_random_real_sample(self, batch_size, seq_len):
        samples = [self.__get_random_sample(seq_len) for _ in range(batch_size)]
        return torch.tensor(samples)

    def __get_random_sample(self, seq_len):
        max_len = self.__len__() - seq_len - 1
        rand = randint(0, max_len)
        return self.examples[rand:rand+seq_len]

