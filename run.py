from data.Dataset import CodeDataset, TextDataset
from models.Discriminator import CNNDiscriminator, Discriminator
from models.Generator import GeneratorLSTM
from train.Trainer import Trainer
from utils.Tokenizer import CodeTokenizerResolver, SentencepieceResolver

training_data = "./demo_code/out_jokes.py"
block_size = 1  # size of training data blocks
sequence_length = 40  # length of generated sequences
embedding_dim = 8  # size of the word vectors in the lookup table - indices are converted to the embedding_dim



if __name__ == '__main__':

    # init tokenizer
    print("Init tokenizer ...")

    special_tokens = [
        '<BOF>',
        '<EOF>',
        '<COMMENT>',
        '<STR_LIT>',
        '<INT_LIT>'
    ]

    tokenizer = SentencepieceResolver(path=training_data, vocab_size=1000, special_tokens=special_tokens)

    # init dataset
    print("Init dataset ... ")

    with open(training_data) as f:
        content = "".join(f.readlines())

    dataset = TextDataset(inp=tokenizer.encode(content), block_size=block_size)

    # init generator
    n_vocab = tokenizer.vocab_size

    generator = GeneratorLSTM(n_vocab=n_vocab, embedding_dim=embedding_dim, hidden_dim=128, num_layers=1)
    print(f"Generator device: {generator.device}")
    discriminator = CNNDiscriminator(n_vocab, 1)

    # trainer
    trainer = Trainer(generator=generator, discriminator=discriminator, sequence_length=sequence_length,
                      dataset=dataset, batch_size=128, max_epochs=2, lr=0.01, nadv_steps=1000, tokenizer=tokenizer,
                      test_file=training_data)

    l, l1, l2 = trainer.train(pretrain_epochs=0)




