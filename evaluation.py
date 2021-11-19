import math

import jellyfish
import torch
from datasets import tqdm
from rouge import Rouge
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, GPT2Tokenizer, AutoModelForCausalLM
import numpy as np
import wandb
import pandas as pd

from baseline import LSTM
from config import init_config
from data.Dataset import TextDataset
from models.generator.TransformerGenerator import PretrainedGPTGenerator

criterion = nn.CrossEntropyLoss()

rouge = Rouge()
models = [
    {
        "name": "baseline 1",
        "description": "gpt2 pretrained for 1 epoch",
        "model_name": "mrchtr/code-gan/gpt-pretrain:v12",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "baseline 2",
        "description": "gpt2 pretrained for 20 epochs",
        "model_name": "mrchtr/code-gan/gpt-pretrain:v30",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "cnn/rsgan",
        "description": "CNN/RSGAN (#1)",
        "model_name": "mrchtr/code-gan/model:v88",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "cnn/wgan-gp",
        "description": "CNN/WGAN-GP (#2)",
        "model_name": "mrchtr/code-gan/model:v87",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "bert/rsgan",
        "description": "BERT/RSGAN (#3)",
        "model_name": "mrchtr/code-gan/model:v73",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "bert/wgan-gp",
        "description": "BERT/WGAN-GP (#4)",
        "model_name": "mrchtr/code-gan/model:v74",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "bert/wgan-gp/random",
        "description": "BERT/WGAN-GP (#4 Random)",
        "model_name": "mrchtr/code-gan/model:v90",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    # open end generation : 0.5 x context size
    {
        "name": "baseline 1",
        "description": "gpt2 pretrained for 1 epoch",
        "model_name": "mrchtr/code-gan/gpt-pretrain:v12",
        "stop_on_line_end": False,
        "sequence_len": 96
    },
    {
        "name": "baseline 2",
        "description": "gpt2 pretrained for 20 epochs",
        "model_name": "mrchtr/code-gan/gpt-pretrain:v30",
        "stop_on_line_end": False,
        "sequence_len": 96
    },
    {
        "name": "cnn/rsgan",
        "description": "CNN/RSGAN (#1)",
        "model_name": "mrchtr/code-gan/model:v88",
        "stop_on_line_end": False,
        "sequence_len": 96
    },
    {
        "name": "cnn/wgan-gp",
        "description": "CNN/WGAN-GP (#2)",
        "model_name": "mrchtr/code-gan/model:v87",
        "stop_on_line_end": False,
        "sequence_len": 96
    },
    {
        "name": "bert/rsgan",
        "description": "BERT/RSGAN (#3)",
        "model_name": "mrchtr/code-gan/model:v73",
        "stop_on_line_end": False,
        "sequence_len": 96
    },
    {
        "name": "bert/wgan-gp",
        "description": "BERT/WGAN-GP (#4)",
        "model_name": "mrchtr/code-gan/model:v74",
        "stop_on_line_end": False,
        "sequence_len": 96
    },
    {
        "name": "bert/wgan-gp/random",
        "description": "BERT/WGAN-GP (#4 Random)",
        "model_name": "mrchtr/code-gan/model:v90",
        "stop_on_line_end": False,
        "sequence_len": 96
    },
    ##### different sequence lengths: 1x context size
    {
        "name": "baseline 2",
        "description": "gpt2 pretrained for 20 epochs",
        "model_name": "mrchtr/code-gan/gpt-pretrain:v30",
        "stop_on_line_end": False,
        "sequence_len": 128
    },
    {
        "name": "cnn/wgan-gp",
        "description": "CNN/WGAN-GP (#2)",
        "model_name": "mrchtr/code-gan/model:v87",
        "stop_on_line_end": False,
        "sequence_len": 128
    },
    {
        "name": "bert/wgan-gp",
        "description": "BERT/WGAN-GP (#4)",
        "model_name": "mrchtr/code-gan/model:v74",
        "stop_on_line_end": False,
        "sequence_len": 128
    },
    {
        "name": "bert/wgan-gp/random",
        "description": "BERT/WGAN-GP (#4 Random)",
        "model_name": "mrchtr/code-gan/model:v90",
        "stop_on_line_end": False,
        "sequence_len": 128
    },
    ##### different sequence lengths: 1.5x context size
    {
        "name": "baseline 2",
        "description": "gpt2 pretrained for 20 epochs",
        "model_name": "mrchtr/code-gan/gpt-pretrain:v30",
        "stop_on_line_end": False,
        "sequence_len": 160
    },
    {
        "name": "cnn/wgan-gp",
        "description": "CNN/WGAN-GP (#2)",
        "model_name": "mrchtr/code-gan/model:v87",
        "stop_on_line_end": False,
        "sequence_len": 160
    },
    {
        "name": "bert/wgan-gp",
        "description": "BERT/WGAN-GP (#4)",
        "model_name": "mrchtr/code-gan/model:v74",
        "stop_on_line_end": False,
        "sequence_len": 160
    },
    {
        "name": "bert/wgan-gp/random",
        "description": "BERT/WGAN-GP (#4 Random)",
        "model_name": "mrchtr/code-gan/model:v90",
        "stop_on_line_end": False,
        "sequence_len": 160
    }
]

models = [
{
        "name": "baseline lstm",
        "description": "lstm",
        "model_name": "mrchtr/code-gan/model:v116",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
{
        "name": "baseline 1",
        "description": "gpt2 pretrained for 1 epoch",
        "model_name": "mrchtr/code-gan/gpt-pretrain:v12",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "baseline 2",
        "description": "gpt2 pretrained for 20 epochs",
        "model_name": "mrchtr/code-gan/gpt-pretrain:v30",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "cnn/rsgan",
        "description": "CNN/RSGAN (#1)",
        "model_name": "mrchtr/code-gan/model:v88",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "cnn/wgan-gp",
        "description": "CNN/WGAN-GP (#2)",
        "model_name": "mrchtr/code-gan/model:v87",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "bert/rsgan",
        "description": "BERT/RSGAN (#3)",
        "model_name": "mrchtr/code-gan/model:v73",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "bert/wgan-gp",
        "description": "BERT/WGAN-GP (#4)",
        "model_name": "mrchtr/code-gan/model:v74",
        "stop_on_line_end": True,
        "sequence_len": 128
    },
    {
        "name": "bert/wgan-gp/random",
        "description": "BERT/WGAN-GP (#4 Random)",
        "model_name": "mrchtr/code-gan/model:v90",
        "stop_on_line_end": True,
        "sequence_len": 128
    }
]


def tokenize_files(source, tokenizer, config):
    with open(source) as f:
        content = "".join(f.readlines())

        tokenized_data = []
        print(f"Content len: {len(content)}")
        # mini_batch = 30000
        # for i in tqdm(range(0, len(content), mini_batch)):
        tokenized_data = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))

        # count iteration
        print(f"Iterations needed: {len(tokenized_data) / config.block_size / config.batch_size}")

        examples = []
        for i in range(0, len(tokenized_data) - config.block_size + 1,
                       config.block_size):  # Truncate in block of block_size
            examples.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_data[i: i + config.block_size])
            )
    return examples


def cos_sim_2d(x, y):
    norm_x = x / np.linalg.norm(x, axis=1, keepdims=True)
    norm_y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.matmul(norm_x, norm_y.T)


def get_ppl(encodings, config, model):
    max_length = config.sequence_length
    stride = config.start_sequence_len

    nlls = []
    for i in range(0, encodings.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(config.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model.step_forward_gumbel(input_ids, return_dict=True, gumbel_forward=False)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)
    return torch.exp(torch.stack(nlls).sum() / end_loc)


def run_evaluation(logger, config, evaluation, tokenizer, dataset, bert, stop_on_line_end=True, break_at=500):
    iteration = 0
    prefix = "evaluation"
    model_name = evaluation['model_name']
    run_name = evaluation['name']
    description = evaluation['description']
    model_name = evaluation['model_name']
    dataloader = DataLoader(dataset, config.batch_size, drop_last=True)
    print(f"++++ Run evaluation for \'{run_name}\' ++++")

    # load model
    if model_name == "mrchtr/code-gan/model:v116":
        generator = LSTM(len(tokenizer), config.eos_token_id, config)
        artifact = logger.use_artifact(model_name, type='model')
        artifact_dir = artifact.download()
        generator.load_state_dict(torch.load(artifact_dir + '/lstm.pth', map_location=torch.device(config.device)))
        generator = generator.to(config.device)
        generator.eval()
    else:
        generator = PretrainedGPTGenerator(config, bos_token=config.eos_token_id)
        artifact = logger.use_artifact(model_name, type='model')
        artifact_dir = artifact.download()
        generator.load_state_dict(torch.load(artifact_dir + '/generator.pth', map_location=torch.device(config.device)))
        generator = generator.to(config.device)
        generator.eval()

    m_levenstein = []
    m_nll = []
    m_ppl = []
    m_rouge_f = []
    m_rouge_r = []
    m_rouge_p = []
    m_cos_sim = []
    m_generation_len = []
    accuray = []

    # run
    dataloader_iter = iter(dataloader)
    for j in range(0, break_at):
        batch = next(dataloader_iter)
        batch = batch.to(config.device)  # <context, ground_truth>
        input = batch[..., :config.start_sequence_len]  # <context>
        generated = generator.gen_sample(input, config.sequence_length, config.batch_size, forward_gumbel=False,
                                         is_eval=stop_on_line_end)

        generated_tensor = generated[..., config.start_sequence_len:]
        ground_truth_tensor = batch[..., config.start_sequence_len:]

        # token accuracy
        generated_flatten = generated_tensor[..., 0:1].to('cpu').numpy().flatten()
        ground_truth_flatten = ground_truth_tensor[..., 0:1].to('cpu').numpy().flatten()
        accuray.append(np.sum(generated_flatten == ground_truth_flatten) / len(generated_flatten))

        generated = generated[..., config.start_sequence_len:].to('cpu').numpy().tolist()  # <generated_tokens>
        ground_truth = batch[..., config.start_sequence_len:].to('cpu').numpy().tolist()  # <ground_truth>
        ground_truth_len = config.sequence_length - config.start_sequence_len
        hypothesis = []
        for _ground_truth in ground_truth:
            if stop_on_line_end and config.eos_token_id in _ground_truth:
                index = _ground_truth.index(config.eos_token_id) + 1  # get index of <EOL> token
                _ground_truth = _ground_truth[0:index] + [config.pad_token_id] * (
                        ground_truth_len - index)  # slicing after + fill of with <EOL> tokens
            hypothesis.append(_ground_truth)

        generated_data_str = tokenizer.batch_decode(generated,
                                                    skip_special_tokens=False)
        real_data_str = tokenizer.batch_decode(hypothesis, skip_special_tokens=False)

        # rougeL & levenstein
        _levenstein = []
        _rouge_f = []
        _rouge_r = []
        _rouge_p = []
        for i in range(0, len(generated_data_str)):
            generated = generated_data_str[i]
            real = real_data_str[i]
            generated = generated.replace("<pad>", "").strip()
            real = real.replace("<pad>", "").strip()
            _levenstein.append(jellyfish.levenshtein_distance(generated, real) / max(len(real), len(generated)))

            if len(generated) > 0 and len(real) > 0:
                score_dict = rouge.get_scores(real, generated)
                _rouge_r.append(score_dict[0]['rouge-l']['r'])
                _rouge_f.append(score_dict[0]['rouge-l']['f'])
                _rouge_p.append(score_dict[0]['rouge-l']['p'])

        # perplexity
        if model_name == "mrchtr/code-gan/model:v116":
            state_h, state_c = generator.init_state(config.start_sequence_len)
            state_h = state_h.to(config.device)
            state_c = state_c.to(config.device)
            x = batch[..., :config.start_sequence_len].to(config.device)
            y = batch[..., 1:config.start_sequence_len + 1].to(config.device)
            y_pred, (state_h, state_c) = generator(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)
            ppl = torch.exp(loss)
            nll = math.log(ppl)

        else:
            ppl = get_ppl(input, config, generator)
            nll = math.log(ppl)
        # bert embeddings
        generated_embed = bert.roberta(generated_tensor).last_hidden_state
        ground_truth_embed = bert.roberta(ground_truth_tensor).last_hidden_state
        cosinus_sim = torch.cosine_similarity(generated_embed.unsqueeze(0), ground_truth_embed.unsqueeze(0))
        cosinus_sim = torch.mean(cosinus_sim)
        generation_len = []
        for generated in generated_data_str:
            generation_len.append(len(generated.replace("<pad>", "").strip()))

        logger.log({
            f"{prefix}/levenstein": np.mean(_levenstein),
            f"{prefix}/edit-similarity": (100 - (100 * np.mean(_levenstein))),
            f"{prefix}/rouge-r": np.mean(_rouge_r),
            f"{prefix}/rouge-f": np.mean(_rouge_f),
            f"{prefix}/rouge-p": np.mean(_rouge_p),
            f"{prefix}/cos-sim": cosinus_sim.item(),
            f"{prefix}/generation-len": np.mean(generation_len)
        })

        m_ppl.append(ppl.item())
        m_nll.append(nll)
        m_levenstein.append(np.mean(_levenstein))
        m_rouge_r.append(np.mean(_rouge_r))
        m_rouge_f.append(np.mean(_rouge_f))
        m_rouge_p.append(np.mean(_rouge_p))
        m_cos_sim.append(cosinus_sim.item())
        m_generation_len.extend(generation_len)

    # Levenstein (%)	Edit Similarity (%)	NNL	PPL	P	R	F	Cosinus Sim
    print(f"Results of {model_name}: ")
    print(
        f"{100 - 100 * np.mean(m_levenstein)};{np.log(np.mean(m_ppl))};{np.mean(m_ppl)};{np.mean(m_rouge_p)};{np.mean(m_rouge_r)};{np.mean(m_rouge_f)};{np.mean(m_cos_sim)};{np.mean(m_generation_len)}")

    data = [[s] for s in m_generation_len]
    table = wandb.Table(data=data, columns=["sequence_length"])
    logger.log(
        {'sequence-length-historgram': wandb.plot.histogram(table, "sequence lengths", "Predicted sequences lengths")}
    )

    logger.log({"eval/nll": np.mean(m_nll),
                "eval/ppl": np.mean(m_ppl),
                "eval/levenstein": np.mean(m_levenstein),
                "eval/rouge-r": np.mean(m_rouge_r),
                "eval/rouge-f": np.mean(m_rouge_f),
                "eval/rouge-p": np.mean(m_rouge_p),
                "eval/cos-sim": np.mean(m_cos_sim)
                })

    return_dict = dict(note=run_name, description=description, model_name=model_name, stop_on_line_end=stop_on_line_end,
                nll=np.mean(m_nll), ppl=np.mean(m_ppl), levenstein=np.mean(m_levenstein),
                edit_sim=(100 - 100 * np.mean(m_levenstein)),
                rouge_r=np.mean(m_rouge_r), rouge_f=np.mean(m_rouge_f), rouge_p=np.mean(m_rouge_p),
                cos_sim=np.mean(m_cos_sim),
                seq_len=np.mean(np.mean(m_generation_len)),
                accuray=(np.mean(accuray))
                )
    print(return_dict)
    return return_dict


if __name__ == '__main__':
    """
    For all tests:
        - init tokenizer
        - init eval dataset
        - init bert model
    """

    # init config
    config = init_config()
    logger = wandb.init(project=config.project_name, config=config)

    results = []
    for evaluation in tqdm(models):
        # run config
        config.sequence_length = evaluation['sequence_len']
        config.block_size = evaluation['sequence_len']
        stop_on_line_end = evaluation['stop_on_line_end']
        break_at = config.eval_break_at

        # init tokenizer
        tokenizer = GPT2Tokenizer(vocab_file="code-tokenizer-vocab.json", merges_file="code-tokenizer-merges.txt")
        tokenizer.add_tokens(config.special_tokens)
        config.vocab_size = len(tokenizer)

        # init bert model for similarity embeddings
        artifact = logger.use_artifact(config.base_bert_model, type='model')
        artifact_dir = artifact.download()
        bert_model = AutoModelForCausalLM.from_pretrained("huggingface/CodeBERTa-small-v1")
        bert_model.resize_token_embeddings(len(tokenizer))
        bert_model = bert_model.to(config.device)
        bert_model.load_state_dict(
            torch.load(artifact_dir + '/code-bert.pth', map_location=torch.device(config.device)))
        bert_model = bert_model.to(config.device)

        # load eval dataset
        config.eos_token_id = tokenizer.encode("<EOL>")[0]
        config.pad_token_id = tokenizer.encode("<pad>")[0]
        eval_data = tokenize_files(config.validation_data, tokenizer, config)
        eval = TextDataset(inp=eval_data, block_size=config.block_size, eos_token_id=config.eos_token_id,
                           pad_token_id=config.pad_token_id)

        results.append(
            run_evaluation(logger, config, evaluation, tokenizer, eval, bert_model, stop_on_line_end=stop_on_line_end,
                           break_at=break_at))

    df = pd.DataFrame(results)
    df.to_csv('evaluation-result.csv')
    wandb.save('evaluation-result.csv')
