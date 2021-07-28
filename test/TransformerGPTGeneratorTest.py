import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from config import init_config
from data.Dataset import CodeDataset, TextDataset
from models.generator.TransformerGenerator import PretrainedGPTGenerator
from utils.Tokenizer import CodeTokenizerResolver, SentencepieceResolver
from torch.utils.data.sampler import SubsetRandomSampler

training_data = "../demo_code/out_jokes.py"
block_size = 32
vocab_size = 1000
batch_size = 64

ntokens = vocab_size
emsize = 768  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 12  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value

if __name__ == '__main__':
    config = init_config()

    print("Starting testing Transformer Generator")

    tokenizer = CodeTokenizerResolver(config=config)

    # initialize dataset
    with open(config.training_data) as f:
        content = "".join(f.readlines())

    # tokenize text - to reduce memory size mini batches will be proceeded
    print(f"Start tokenization of training data ...")
    tokenized_training_data = []
    mini_batch = 500
    for i in tqdm(range(0, len(content), mini_batch)):
        tokenized_training_data += tokenizer.encode(content[i:i + mini_batch])

    dataset = TextDataset(inp=tokenized_training_data, block_size=config.block_size)

    assert tokenizer.vocab_size == config.vocab_size

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    x, y = dataset.__getitem__(42)
    print(f"X: {x}")
    print(f"Y: {y} --> same as x shifted one to the right")

    model = PretrainedGPTGenerator(config)
    #src_mask = model.init_state(block_size)
    #model.forward(x, src_mask)  # takes input and src_mask

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
    for epoch in range(100):
        src_mask = model.init_state(batch_size)
        model.train()
        i = 0
        losses = []
        for batch, (x, y) in tqdm(enumerate(train_loader), ncols=75):
            i += 1
            pred, _, next_token = model(x)
            y = y[:, -1]
            loss = criterion(pred, y.view(-1))  # do it on gumbel_t instead of prediction
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())
            if i % 10 == 0:
                print(f"Loss epoch {np.mean(losses)}")
                losses = []
                sample = model.sample(x, 40, batch_size, num_samples=1)
                print(f"Prediction {tokenizer.decode(sample.numpy()[0].tolist())}")






