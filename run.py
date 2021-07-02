from data.CodeDataset import CodeDataset
from models.Discriminator import CNNDiscriminator, Discriminator
from models.Generator import GeneratorLSTM
from train.Trainer import Trainer
from utils.Tokenizer import CodeTokenizerResolver

training_data = "./demo_code/"
block_size = 1  # size of training data blocks
sequence_length = 8  # length of generated sequences
embedding_dim = 32  # size of the word vectors in the lookup table - indices are converted to the embedding_dim

def sample_print(tokenizer):
    sample = "class Superman:$    def __init__(self):"
    tokenized = tokenizer.tokenize(sample)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized)
    output = tokenizer.convert_ids_to_tokens(input_ids)
    decoded = tokenizer.convert_tokens_to_string(output)

    print(f"""
    tokenized: {tokenized} \n
    input_ids: {input_ids} \n
    output: {output} \n
    decoded {decoded}
            """)

if __name__ == '__main__':
    # init tokenizer
    print("Init tokenizer ...")
    resolver = CodeTokenizerResolver(training_files=training_data, path="./.checkpoints")
    tokenizer = resolver.tokenizer

    print(f"Test tokenizer for 'def' : {tokenizer.convert_tokens_to_ids('def')}")

    # init dataset
    print("Init dataset ... ")
    dataset = CodeDataset(root_dir="demo_code", tokenizer=tokenizer, block_size=block_size)

    sample_print(dataset.tokenizer)

    # init generator
    n_vocab = dataset.vocab_size()


    generator = GeneratorLSTM(n_vocab=n_vocab, embedding_dim=embedding_dim, hidden_dim=128, num_layers=1)
    print(f"Generator device: {generator.device}")
    discriminator = CNNDiscriminator(n_vocab, 1)

    # trainer
    trainer = Trainer(generator=generator, discriminator=discriminator, sequence_length=sequence_length, dataset=dataset, batch_size=128, max_epochs=5000, lr=0.0002)

    trainer.train(pretrain_epochs=2)


