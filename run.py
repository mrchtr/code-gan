from data.CodeDataset import CodeDataset
from models.Discriminator import CNNDiscriminator, Discriminator
from models.Generator import GeneratorLSTM
from train.Trainer import Trainer
from utils.Tokenizer import CodeTokenizerResolver

training_data = "./demo_code/"
block_size = 32
sequence_length = 32  # length of generated sequences
out_size = 32
stride = 2

if __name__ == '__main__':
    # init tokenizer
    print("Init tokenizer ...")
    resolver = CodeTokenizerResolver(training_files=training_data, path="./.checkpoints")
    tokenizer = resolver.tokenizer

    print(f"Test tokenizer for 'def' : {tokenizer.convert_tokens_to_ids('def')}")

    # init dataset
    print("Init dataset ... ")
    dataset = CodeDataset(root_dir="demo_code", tokenizer=tokenizer, block_size=block_size)

    # init generator
    n_vocab = dataset.vocab_size()
    embedding_dim = block_size

    generator = GeneratorLSTM(n_vocab=n_vocab, embedding_dim=embedding_dim, hidden_dim=128, num_layers=3)
    discriminator = CNNDiscriminator(n_vocab, 1)

    # trainer
    trainer = Trainer(generator=generator, discriminator=discriminator, sequence_length=sequence_length, dataset=dataset, batch_size=16, max_epochs=5000)

    trainer.train()


