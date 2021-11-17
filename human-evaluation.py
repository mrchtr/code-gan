import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

import wandb
from config import init_config
from data.Dataset import TextDataset
from evaluation import tokenize_files
from models.generator.TransformerGenerator import PretrainedGPTGenerator
from utils.Preprocessor import postprocess

break_at = 25
batch_size = 2
if __name__ == '__main__':

    # initialize
    models = [
        {
            "name": "baseline",
            "model_name": "mrchtr/code-gan/gpt-pretrain:v30"
        },
        {
            "name": "bert/wgan-gp",
            "model_name": "mrchtr/code-gan/model:v74",
        },
        {
            "name": "bert/wgan-gp/random",
            "model_name": "mrchtr/code-gan/model:v90",
        }
    ]

    generators = []

    config = init_config()
    logger = wandb.init(project=config.project_name, config=config)

    # init tokenizer
    tokenizer = GPT2Tokenizer(vocab_file="code-tokenizer-vocab.json", merges_file="code-tokenizer-merges.txt")
    tokenizer.add_tokens(config.special_tokens)
    config.vocab_size = len(tokenizer)

    # load eval dataset
    config.eos_token_id = tokenizer.encode("<EOL>")[0]
    config.pad_token_id = tokenizer.encode("<pad>")[0]
    eval_data = tokenize_files(config.validation_data, tokenizer, config)
    eval = TextDataset(inp=eval_data, block_size=config.block_size, eos_token_id=config.eos_token_id,
                       pad_token_id=config.pad_token_id)



    # for each model generated samples of specific size
    sizes = [32, 64, 96]
    sequence_lenghts = [x + config.start_sequence_len for x in sizes]

    dataloader = DataLoader(eval, batch_size, drop_last=True, shuffle=True)

    results = []
    # initialize generators
    for model in models:
        model_name = model['model_name']
        generator = PretrainedGPTGenerator(config, bos_token=config.eos_token_id)
        artifact = logger.use_artifact(model_name, type='model')
        artifact_dir = artifact.download()
        generator.load_state_dict(torch.load(artifact_dir + '/generator.pth', map_location=torch.device(config.device)))
        generator = generator.to(config.device)
        generator.eval()

        for sequence_lenght in sequence_lenghts:

            dataloader_iter = iter(dataloader)
            for j in range(0, break_at):

                batch = next(dataloader_iter)
                batch = batch.to(config.device)  # <context, ground_truth>
                input = batch[..., :config.start_sequence_len]  # <context>

                generated = generator.gen_sample(input, sequence_lenght, batch_size, forward_gumbel=False,
                                                 is_eval=False)

                context = batch[..., :config.start_sequence_len].to('cpu').numpy().tolist()
                generated = generated[..., config.start_sequence_len:].to('cpu').numpy().tolist()  # <generated_tokens>

                context_str = tokenizer.batch_decode(context, skip_special_tokens=False)
                generated_data_str = tokenizer.batch_decode(generated,
                                                            skip_special_tokens=False)

                ground_truth_data_str = tokenizer.batch_decode(batch,
                                                            skip_special_tokens=False)

                for i in range(len(context_str)):
                    condition = context_str[i]
                    ground_truth = ground_truth_data_str[i]
                    ground_truth = postprocess(ground_truth)
                    sample = context_str[i] + generated_data_str[i]
                    condition = postprocess(condition)
                    sample = postprocess(sample)
                    results.append({
                        'model': model_name,
                        'sequence_length': sequence_lenght,
                        'condition' : condition,
                        'ground truth': ground_truth,
                        'sample': postprocess(sample)
                    })

    df = pd.DataFrame(results)
    df.to_csv('human-eval-result.csv')
    wandb.save('human-eval-result.csv')