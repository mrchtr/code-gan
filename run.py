from numpy import load

from config import init_config
from data.Dataset import TextDataset
from models.discriminator.Discriminator import CNNDiscriminator, CodeBertDiscriminator
from models.generator.Generator import GeneratorLSTM

from models.generator.TransformerGenerator import TransformerGenerator, PretrainedGPTGenerator
from train.Trainer import Trainer
from utils.Tokenizer import CodeTokenizerResolver, SentencepieceResolver
import os
from tqdm import tqdm
import wandb

def init_wandb_logger(config):
    project_name = "code-gan-debug"
    return wandb.init(project=project_name, config=config)

def tokenize_files(source, tokenizer):
    with open(source) as f:
        content = "".join(f.readlines())

        tokenized_data = []
        mini_batch = 500
        for i in tqdm(range(0, len(content), mini_batch)):
            tokenized_data += tokenizer.encode(content[i:i + mini_batch]).ids
    return tokenized_data

def load_datasets(config, tokenizer):
    training_data = tokenize_files(config.training_data, tokenizer)
    train = TextDataset(inp=training_data, block_size=config.block_size)

    eval_data = tokenize_files(config.validation_data, tokenizer)
    eval = TextDataset(inp=eval_data, block_size=config.block_size)
    return train, eval

if __name__ == '__main__':

    config = init_config()

    if config.debug:
        os.environ["WANDB_MODE"] = "offline"

    logger = init_wandb_logger(config)

    print("Start Code-GAN training with the following configuration: ")
    print(f"Generator: {config.generator}")
    print(f"Discriminator: {config.discriminator}")
    print(60 * "-")

    # initialize tokenizer
    tokenizer = CodeTokenizerResolver(config=config).get()
    train, eval = load_datasets(config, tokenizer)

    generator = PretrainedGPTGenerator(config)
    if config.discriminator == "CNN":
        discriminator = CNNDiscriminator(config)
    else:
        discriminator = CodeBertDiscriminator()

    trainer = Trainer(generator, discriminator, train, tokenizer, config, logger=logger)
    trainer.train()

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('generator.pth')
    logger.log_artifact(artifact)

    logger.finish()




