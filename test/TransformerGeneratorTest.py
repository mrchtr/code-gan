import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from data.Dataset import CodeDataset, TextDataset
from models.generator.TransformerGenerator import TransformerGenerator
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
model = TransformerGenerator(ntokens, emsize, nhead, nhid, nlayers, dropout)

if __name__ == '__main__':
    print("Starting testing Transformer Generator")

    tokenizer = SentencepieceResolver(path=training_data, vocab_size=vocab_size)

    # init dataset
    print("Init dataset ... ")

    with open(training_data) as f:
        content = "".join(f.readlines())

    dataset = TextDataset(inp=tokenizer.encode(content), block_size=block_size)

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    x, y = dataset.__getitem__(42)
    print(f"X: {x}")
    print(f"Y: {y} --> same as x shifted one to the right")

    src_mask = model.generate_square_subsequent_mask(block_size)
    model.forward(x, src_mask)  # takes input and src_mask

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
    for epoch in range(100):
        src_mask = model.generate_square_subsequent_mask(batch_size)
        model.train()
        i = 0
        losses = []
        for batch, (x, y) in tqdm(enumerate(train_loader), ncols=75):
            i += 1
            output, pred, next_token = model(x, src_mask)
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






