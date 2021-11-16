import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
from config import init_config
from run import init_wandb_logger, load_datasets


class LSTM(nn.Module):
    """
    prediction can be done as follow:
            prediction = lstm.gen_sample(x, config.sequence_length)
            tokenizer.batch_decode(prediction,
                                   skip_special_tokens=False)
    """
    def __init__(self, n_vocab, eos_token, config):
        super(LSTM, self).__init__()
        self.eos_token = eos_token
        self.config = config
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

    def gen_sample(self, context, sequence_length, batch_size, forward_gumbel=True, is_eval=False):
        sequence_length = sequence_length - self.config.start_sequence_len
        prediction = []
        for batch in context:
            tokens = batch.numpy()
            for i in range(0, sequence_length):
                state_h, state_c = self.init_state(len(batch))
                x = torch.tensor(tokens[-self.config.start_sequence_len:]).to(self.config.device)
                y_pred, (state_h, state_c) = self(x[None, :], (state_h, state_c))
                last_word_logits = y_pred[0][-1]
                p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
                token = np.random.choice(len(last_word_logits), p=p)
                tokens = np.append(tokens, token)

                if is_eval and token == self.eos_token:
                    break
            prediction.append(torch.tensor(tokens))



        if is_eval:
            # pad all seqs to desired length
            out_tensor = prediction[0].data.new(*(batch_size, self.config.sequence_length)).fill_(self.config.pad_token_id)
            for i, tensor in enumerate(prediction):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            out_tensor = out_tensor.to(self.config.device)
            return out_tensor

        return torch.stack(prediction)

if __name__ == '__main__':


    config = init_config()
    logger = init_wandb_logger(config)

    print(60 * "-")
    print("Run training for baseline model.")
    print(f"GPU available : {config.device}")
    print(f"Debugging on : {config.debug}")
    print(60 * "-")

    # Load tokenizer
    tokenizer = GPT2Tokenizer(vocab_file="code-tokenizer-vocab.json", merges_file="code-tokenizer-merges.txt")
    tokenizer.add_tokens(config.special_tokens)
    config.vocab_size = len(tokenizer)

    config.eos_token_id = tokenizer.encode("<EOL>")[0]
    config.pad_token_id = tokenizer.encode("<pad>")[0]
    train, eval = load_datasets(config, tokenizer, config.eos_token_id, config.pad_token_id)

    # initialize model
    lstm = LSTM(config.vocab_size)
    lstm = lstm.to(config.device)
    lstm.train()

    dataloader = DataLoader(train, batch_size=config.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)



    for epoch in range(config.baseline_train_epochs):
        state_h, state_c = lstm.init_state(config.start_sequence_len)
        state_h = state_h.to(config.device)
        state_c = state_c.to(config.device)
        for batch, sample in tqdm(enumerate(dataloader)):
            # prepare input
            x = sample[..., :config.start_sequence_len].to(config.device)
            y = sample[..., 1:config.start_sequence_len+1].to(config.device)

            optimizer.zero_grad()
            y_pred, (state_h, state_c) = lstm(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()
            logger.log({"pretrain/loss": loss.item()})
            #print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})

        torch.save(lstm.state_dict(), 'lstm.pth')
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file('lstm.pth')
        logger.log_artifact(artifact)