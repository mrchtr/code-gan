from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from data.CodeDataset import CodeDataset
from models.Generator import GeneratorLSTM
from utils.Tokenizer import CodeTokenizerResolver

if __name__ == '__main__':
    batch_size = 256
    seq_len = 60
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

    prediction, state, _, _ = generator.forward(x.unsqueeze(0), hidden)
    print(f"X : {x}")
    print(f"Y': {prediction}")

    print("Test training ... ")

    dataloader = DataLoader(dataset, batch_size, drop_last=True, shuffle=True)

    generator.train()

    criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(generator.parameters(), lr=0.001)
    for epoch in range(10):
        hidden = generator.init_state(batch_size)
        for batch, (x, y) in tqdm(enumerate(dataloader), ncols=75):

            pred, hidden, next_token, gumbel_t = generator(x, hidden)
            loss = criterion(pred.transpose(1, 2), y) # do it on gumbel_t instead of prediction

            optimizer.zero_grad()
            hidden = hidden[0].detach(), hidden[1].detach()
            loss.backward()
            optimizer.step()

            # predict
            if batch % 100 == 0:
                p = prediction.detach().numpy()
                token = np.random.choice(dataset.vocab_size(), p=p[0][-1])
                sample = np.append(x[0].detach().numpy(), token)
                sample = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sample))
                y_1 = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y[0]))
                print(f"Sample: {sample} / Should be {y_1}")
                print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})





    print("done")




