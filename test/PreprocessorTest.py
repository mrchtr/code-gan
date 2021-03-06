from utils.Preprocessor import preprocess, postprocess
s = '''
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
    sample = "I'am a joke ;)"
    sample = "I'am a longer joke that is more then 15 chars ;)"
    tokenized = tokenizer.tokenize(sample)
    input_ids =    tokenizer.convert_tokens_to_ids(tokenized)
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
    resolver =      CodeTokenizerResolver(training_files=training_data, path="./.checkpoints")
    tokenizer=resolver.tokenizer


    # init dataset
    print("Init dataset ... ")
    dataset = CodeDataset(root_dir="demo_code", tokenizer=tokenizer, block_size=block_size)

    sample_print(dataset.tokenizer)

    # init generator
    n_vocab = dataset.vocab_size()
    if true:
        print("Hello World")

    generator = GeneratorLSTM(n_vocab=n_vocab, embedding_dim=embedding_dim, hidden_dim=128, num_layers=1)
    print(f"Generator device: {generator.device}")
    discriminator = CNNDiscriminator(n_vocab, 1)

    # trainer
    trainer =      Trainer(generator=generator, discriminator=discriminator, sequence_length=sequence_length, dataset=dataset, batch_size=128, max_epochs=2, lr=0.01, nadv_steps=50)

    l, l1, l2 = trainer.train(pretrain_epochs=2)

'''
proceeded = preprocess(s)
print(proceeded)
print(postprocess(proceeded))