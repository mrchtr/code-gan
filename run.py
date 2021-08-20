from datasets import load_dataset
from numpy import load
from transformers import AutoTokenizer

from config import init_config
from data.Dataset import TextDataset
from models.discriminator.Discriminator import CNNDiscriminator
from models.generator.Generator import GeneratorLSTM

from models.generator.TransformerGenerator import TransformerGenerator, PretrainedGPTGenerator
from train.Trainer import Trainer
from utils.Tokenizer import CodeTokenizerResolver, SentencepieceResolver
import os
from tqdm import tqdm
import wandb

project_name = "code-gan-debug"

#os.environ["WANDB_MODE"] = "offline"

def init_wandb_logger(config):
    return wandb.init(project=project_name, config=config)

if __name__ == '__main__':
    config = init_config()
    logger = init_wandb_logger(config)

    print("Start Code-GAN training with the following configuration: ")
    print(f"Generator {config.generator}")
    print(f"Discriminator {config.discriminator}")

    # initialize tokenizer
    if config.generator == "GPTCode":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2")
        #tokenizer.pad_token_id = tokenizer.eos_token_id
        config.vocab_size = len(tokenizer)
    else:
        tokenizer = CodeTokenizerResolver(config=config)



    # tokenize text - to reduce memory size mini batches will be proceeded
    if config.benchmark_dataset == True:
        loaded = load(config.training_data)
        dataset = TextDataset(inp=loaded, block_size=config.block_size)
    else:
        print(f"Start tokenization of training data ...")
        # initialize dataset
        with open(config.training_data) as f:
            content = "".join(f.readlines())

        tokenized_training_data = []
        mini_batch = 500
        for i in tqdm(range(0, len(content), mini_batch)):
            tokenized_training_data += tokenizer.encode(content[i:i+mini_batch])

        dataset = TextDataset(inp=tokenized_training_data, block_size=config.block_size)

        # creating reference dataset for evaluation
        print(f"Start tokenization of evaluation data ...")
        with open(config.validation_data) as f:
            content = "".join(f.readlines())

        tokenized_reference_data = []
        mini_batch = 500
        for i in tqdm(range(0, len(content), mini_batch)):
            tokenized_reference_data += tokenizer.encode(content[i:i + mini_batch])

        reference_data = TextDataset(inp=tokenized_reference_data, block_size=config.block_size)

        #assert len(tokenizer) == config.vocab_size

    # initialize generator model
    if config.generator == "Transformer":
        generator = TransformerGenerator(config)
    elif config.generator == "LSTM":
        generator = GeneratorLSTM(config)
    elif config.generator == "GPTCode":
        generator = PretrainedGPTGenerator(config)
        print(generator)
    else:
        raise Exception(f"Can't create unknown generator {config.generator}")

    # initialize discriminator model
    if config.discriminator == "CNN":
        discriminator = CNNDiscriminator(config)
    else:
        raise Exception(f"Can't create unknown discriminator {config.discriminator}")

    # trainer
    trainer = Trainer(generator, discriminator, dataset, tokenizer, config, logger=logger)
    trainer.train()

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('generator.pth')
    logger.log_artifact(artifact)

    logger.finish()




