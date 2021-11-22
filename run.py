import argparse
from collections import OrderedDict

import torch
from numpy import load
from torch import nn
from torch.nn import Identity
from transformers import GPT2Tokenizer, RobertaModel, RobertaConfig, BertConfig, BertModel, RobertaForCausalLM, \
    AutoModelForCausalLM, AutoTokenizer

from config import init_config
from data.Dataset import TextDataset
from models.discriminator.Discriminator import CNNDiscriminator, CodeBertDiscriminator, RelGAN_D

from models.generator.TransformerGenerator import PretrainedGPTGenerator
from train.Pretrainer import Pretrainer
from train.Trainer import Trainer
from utils.Tokenizer import CodeTokenizerResolver
import os
from tqdm import tqdm
import wandb

def init_wandb_logger(config):
    return wandb.init(project=config.project_name, config=config)

def tokenize_files(source, tokenizer, config):
    with open(source) as f:
        content = "".join(f.readlines())

        tokenized_data = []
        print(f"Content len: {len(content)}")
        #mini_batch = 30000
        #for i in tqdm(range(0, len(content), mini_batch)):
        tokenized_data = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))
        examples = []
        for i in range(0, len(tokenized_data) - config.block_size + 1, config.block_size):  # Truncate in block of block_size
            examples.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_data[i: i + config.block_size])
            )
    return examples

def load_datasets(config, tokenizer, eos_token_id, pad_token_id):
    training_data = tokenize_files(config.training_data, tokenizer, config)
    train = TextDataset(inp=training_data, block_size=config.block_size, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

    eval_data = tokenize_files(config.validation_data, tokenizer, config)
    eval = TextDataset(inp=eval_data, block_size=config.block_size, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
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
    #tokenizer = CodeTokenizerResolver(config=config).get()


    # Load pretrained model and tokenizer
    if config.saved_model != 'GPT-Code':
        tokenizer = GPT2Tokenizer(vocab_file="code-tokenizer-vocab.json", merges_file="code-tokenizer-merges.txt")
        tokenizer.add_tokens(config.special_tokens)
        config.vocab_size = len(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")
        config.vocab_size = len(tokenizer)

    config.eos_token_id = tokenizer.encode("<EOL>")[0]
    config.pad_token_id = tokenizer.encode("<pad>")[0]
    train, eval = load_datasets(config, tokenizer, config.eos_token_id, config.pad_token_id)
    context, ground_truth = train.get_random_context_with_ground_truth(start_len=10, seq_len=12, batch_size=1)
    print(f"Context: {context}")
    print(f"Ground Truth: {ground_truth[0]}")


    generator = PretrainedGPTGenerator(config, bos_token=config.eos_token_id)
    if config.saved_model and config.saved_model != 'GPT-Code' and config.pretraining_epochs == 0:
        artifact = logger.use_artifact(config.saved_model, type='model')
        artifact_dir = artifact.download()
        generator.load_state_dict(torch.load(artifact_dir + '/generator.pth', map_location=torch.device(config.device)))

    if config.discriminator == "CNN":
        discriminator = RelGAN_D(config)
    elif config.discriminator == "BERT":
        artifact = logger.use_artifact(config.base_bert_model, type='model')
        artifact_dir = artifact.download()
        base_model = AutoModelForCausalLM.from_pretrained("huggingface/CodeBERTa-small-v1")
        base_model.resize_token_embeddings(len(tokenizer))
        base_model = base_model.to(config.device)
        if config.saved_model != 'GPT-Code':
            base_model.load_state_dict(torch.load(artifact_dir + '/code-bert.pth', map_location=torch.device(config.device)))
        discriminator = CodeBertDiscriminator(base_model, config)
    else:
        raise("No valid discriminator defined")

    trainer = Trainer(generator, discriminator, train, tokenizer, config, logger=logger, dataset_eval=eval)
    trainer.train()

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('generator.pth')
    logger.log_artifact(artifact)

    logger.finish()




