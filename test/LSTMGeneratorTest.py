from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from data.Dataset import CodeDataset
from models.Generator import GeneratorLSTM
from utils.Tokenizer import CodeTokenizerResolver
from torch.utils.data.sampler import SubsetRandomSampler

if __name__ == '__main__':
    batch_size = 8
    seq_len = 1
    print("Starting testing LSTM Generator")

    print("Init tokenizer ...")
    resolver = CodeTokenizerResolver(training_files="../demo_code/", path="./.checkpoints")

    print("Init dataset ... ")
    dataset = CodeDataset(root_dir="../demo_code", tokenizer=resolver.tokenizer, block_size=seq_len)
    tokenizer = resolver.tokenizer
    generator = GeneratorLSTM(n_vocab=dataset.vocab_size(), embedding_dim=16, hidden_dim=128)

    x, y = dataset.__getitem__(42)
    print(f"X: {x}")
    print(f"Y: {y} --> same as x shifted one to the right")

    hidden = generator.init_state(batch_size=1)

    print(f"""
        LSTM Input Definition:
        inp : (batch_size, sequence_length, input_dim) -> {x.shape}
        hidden : (n_layers, sequence_length, hidden_dim) -> {hidden[0].shape}
        """)

    prediction, state, _ = generator.forward(x.unsqueeze(0), hidden)
    print(f"X : {x}")
    print(f"Y': {prediction}")

    print("Test training ... ")

    # split train and evaluation
    validation_split = .2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, drop_last=True)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, drop_last=True)



    criterion = nn.NLLLoss()



    optimizer = optim.Adam(generator.parameters(), lr=0.015)
    for epoch in range(100):
        hidden = generator.init_state(batch_size)

        generator.train()
        for batch, (x, y) in tqdm(enumerate(train_loader), ncols=75):

            pred, hidden, next_token = generator(x, hidden)
            loss = criterion(pred, y.view(-1)) # do it on gumbel_t instead of prediction

            optimizer.zero_grad()
            hidden = hidden[0].detach(), hidden[1].detach()
            loss.backward()
            optimizer.step()

        print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
        sample = generator.sample(x, 20, batch_size, num_samples=1)
        for row in sample:
            print(row)
            print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(row)))

        generator.eval()
        val_hidden = generator.init_state(batch_size)
        val_losses = []
        for batch, (x,y) in enumerate(validation_loader):
            out, val_hidden, _ = generator(x, val_hidden)
            val_loss = criterion(out, y.view(-1))
            val_losses.append(val_loss.item())
        print(f"Validation loss: {np.mean(val_losses)}")






    print("done")




