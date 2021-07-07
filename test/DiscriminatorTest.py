from random import randint

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from data.CodeDataset import CodeDataset
from models.Discriminator import Discriminator, CNNDiscriminator
from models.Generator import GeneratorLSTM
from utils.Tokenizer import CodeTokenizerResolver





if __name__ == '__main__':
    seq_len = 1

    print("Discriminator Test")

    print("Init tokenizer ...")
    resolver = CodeTokenizerResolver(training_files="../demo_code/", path="./.checkpoints")

    print("Init dataset ... ")
    dataset = CodeDataset(root_dir="../demo_code", tokenizer=resolver.tokenizer, block_size=1)
    tokenizer = resolver.tokenizer
    discriminator = CNNDiscriminator(dataset.vocab_size(), 1)

    seq_len = 20
    letter = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    random_str = "".join(letter[randint(0, len(letter)-1)] for x in range(0, seq_len))
    tokenized_random_str = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(random_str))).reshape(1, seq_len)

    real_sample = dataset.get_random_real_sample(1, seq_len)

    print(random_str)
    print(tokenized_random_str)
    print(real_sample)

    # sample training loop
    dataloader = DataLoader(dataset, batch_size=32,drop_last=True)
    optimizer = optim.Adam(discriminator.parameters(), lr=0.015)
    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        discriminator.train()
        losses = []
        for batch, _ in enumerate(dataloader):
            real_sample = dataset.get_random_real_sample(32, seq_len)
            random_str = "".join(letter[randint(0, len(letter) - 1)] for x in range(0, seq_len))
            genrated = torch.tensor(
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(random_str))).reshape(1, seq_len)

            d_out_real = discriminator(real_sample)
            d_out_fake = discriminator(genrated)

            loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real))

            discriminator.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print({'epoch': epoch, 'batch': batch, 'loss': np.mean(losses)})