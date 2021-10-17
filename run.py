import argparse

from numpy import load
from transformers import GPT2Tokenizer

from config import init_config
from data.Dataset import TextDataset
from models.discriminator.Discriminator import CNNDiscriminator, CodeBertDiscriminator, RelGAN_D
from models.generator.Generator import GeneratorLSTM

from models.generator.TransformerGenerator import TransformerGenerator, PretrainedGPTGenerator
from train.Pretrainer import Pretrainer
from train.Trainer import Trainer
from utils.Tokenizer import CodeTokenizerResolver
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

def pretrain():
    tokenizer_pretrain = GPT2Tokenizer(vocab_file="./code-tokenizer-vocab.json",
                                       merges_file="./code-tokenizer-merges.txt",
                                       bos_token="<EOL>", eos_token="<EOL>")
    pretrain = Pretrainer('GPT2', tokenizer_pretrain, config)
    pretrain.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main training programm controll')
    parser.add_argument('-p', '--pretrain',
                        action='store_true',
                        dest='pretraining',
                        )

    args = parser.parse_args()

    config = init_config()

    if config.debug:
        os.environ["WANDB_MODE"] = "offline"


    logger = init_wandb_logger(config)

    print(60 * "-")
    print("Start Code-GAN training with the following configuration: ")
    print(f"Generator: {config.generator}")
    print(f"Discriminator: {config.discriminator}")
    print(f"GPU available : {config.device}")
    print(f"Debugging on : {config.debug}")
    print(60 * "-")

    # initialize tokenizer
    tokenizer = CodeTokenizerResolver(config=config).get()

    #if args.pretraining:
    #    print("Start pretraining generator ...")
    #    pretrain()
    #    exit()

    config.eos_token_id = tokenizer.encode("<EOL>").ids[0]
    train, eval = load_datasets(config, tokenizer)
    context, ground_truth = train.get_random_context_with_ground_truth(start_len=10, seq_len=11, batch_size=1)
    print(f"Context: {ground_truth[0][:-1]}")
    print(f"Context: {context}")
    print(f"Labels: {ground_truth[0][1:]}")


    generator = PretrainedGPTGenerator(config, pretrained_model="./gpt2-code-pretrained", bos_token=config.eos_token_id)
    if config.discriminator == "CNN":
        discriminator = RelGAN_D(config)
    else:
        #discriminator = CodeBertDiscriminator()
        None

    trainer = Trainer(generator, discriminator, train, tokenizer, config, logger=logger, dataset_eval=eval)
    trainer.train()

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('generator.pth')
    logger.log_artifact(artifact)

    logger.finish()




