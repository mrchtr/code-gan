import jellyfish
import torch
from datasets import tqdm
from rouge import Rouge
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, GPT2Tokenizer, AutoModelForCausalLM
import numpy as np
import wandb

from config import init_config
from data.Dataset import TextDataset
from models.generator.TransformerGenerator import PretrainedGPTGenerator

rouge = Rouge()
"""
     {
     "id": "pretrain",
     "name": "fearful-phantasm-56",
     "model_name": "mrchtr/code-gan/gpt-pretrain:v30",
     "note": "pretrained baseline gpt2"
 },
 {
     "id": "1",
     "name": "glorious-violet-65",
     "model_name": "mrchtr/code-gan/model:v54",
     "note": "cnn rsgan"
 },
 {
     "id": "2",
     "name": "resilient-brook-66",
     "model_name": "mrchtr/code-gan/model:v53",
     "note": "cnn wgan-gp"
 },
 """
models = [
    {
        "id": "3",
        "name": "scarlet-wildflower-69",
        "model_name": "mrchtr/code-gan/model:v61",
        "note": "bert rsgan"
    },
    {
        "id": "4",
        "name": "dauntless-elevator-70",
        "model_name": "mrchtr/code-gan/model:v60",
        "note": "bert wsgan-gp"
    }
]


def tokenize_files(source, tokenizer, config):
    with open(source) as f:
        content = "".join(f.readlines())

        tokenized_data = []
        print(f"Content len: {len(content)}")
        #mini_batch = 30000
        #for i in tqdm(range(0, len(content), mini_batch)):
        tokenized_data = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))

        # count iteration
        print(f"Iterations needed: {len(tokenized_data) / config.block_size / config.batch_size}")

        examples = []
        for i in range(0, len(tokenized_data) - config.block_size + 1, config.block_size):  # Truncate in block of block_size
            examples.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_data[i: i + config.block_size])
            )
    return examples


def cos_sim_2d(x, y):
    norm_x = x / np.linalg.norm(x, axis=1, keepdims=True)
    norm_y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.matmul(norm_x, norm_y.T)

def run_evaluation(logger, config, evaluation, tokenizer, dataset, bert):
    model_name = evaluation['model_name']
    prefix = f"eval-{evaluation['id']}"
    dataloader = DataLoader(dataset, config.batch_size, drop_last=True)
    print(f"Run evaluation for \'{evaluation['note']}\' ...")

    # load model
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

    # run
    for i, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(config.device) # <context, ground_truth>
        input = batch[...,:config.start_sequence_len]  # <context>
        generated = generator.gen_sample(input, config.sequence_length, config.batch_size, forward_gumbel=False, is_eval=True)

        generated_tensor = generated[...,config.start_sequence_len:]
        ground_truth_tensor = batch[..., config.start_sequence_len:]

        generated = generated[...,config.start_sequence_len:].to('cpu').numpy().tolist()  # <generated_tokens>
        ground_truth = batch[..., config.start_sequence_len:].to('cpu').numpy().tolist()  # <ground_truth>
        ground_truth_len = config.sequence_length - config.start_sequence_len
        hypothesis = []
        for _ground_truth in ground_truth:
            if config.eos_token_id in _ground_truth:
                index = _ground_truth.index(config.eos_token_id) + 1# get index of <EOL> token
                _ground_truth = _ground_truth[0:index] + [config.pad_token_id] * (ground_truth_len - index) # slicing after + fill of with <EOL> tokens
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
            _levenstein.append(jellyfish.levenshtein_distance(generated, real) / max(len(real), len(generated)))



            score_dict = rouge.get_scores(real, generated)
            _rouge_r.append(score_dict[0]['rouge-l']['r'])
            _rouge_f.append(score_dict[0]['rouge-l']['f'])
            _rouge_p.append(score_dict[0]['rouge-l']['p'])

        # perplexity
        loss = generator.step_forward_gumbel(input, return_dict=True, gumbel_forward=False).loss
        nll = loss.mean()
        ppl = torch.exp(loss.mean())

        # bert embeddings
        generated_embed = bert.roberta(generated_tensor).last_hidden_state
        ground_truth_embed = bert.roberta(ground_truth_tensor).last_hidden_state
        cosinus_sim = torch.cosine_similarity(generated_embed.unsqueeze(0), ground_truth_embed.unsqueeze(0))
        cosinus_sim = torch.mean(cosinus_sim)

        logger.log({f"{prefix}/nll": nll.item(),
                    f"{prefix}/ppl": ppl.item(),
                    f"{prefix}/levenstein": np.mean(_levenstein),
                    f"{prefix}/rouge-r": np.mean(_rouge_r),
                    f"{prefix}/rouge-f": np.mean(_rouge_f),
                    f"{prefix}/rouge-p": np.mean(_rouge_p),
                    f"{prefix}/cos-sim": cosinus_sim.item(),
                    })

        m_nll.append(nll.item())
        m_ppl.append(ppl.item())
        m_levenstein.append(np.mean(_levenstein))
        m_rouge_r.append(np.mean(_rouge_r))
        m_rouge_f.append(np.mean(_rouge_f))
        m_rouge_p.append(np.mean(_rouge_p))
        m_cos_sim.append(cosinus_sim.item())

        if i > 1000:
            break


    print(f"NLL: {np.mean(m_nll)}")
    print(f"PPL: {np.mean(m_ppl)}")
    print(f"Levenstein: {np.mean(m_levenstein)}")
    print(f"Rouge-R: {np.mean(m_rouge_r)}")
    print(f"Rouge-F: {np.mean(m_rouge_f)}")
    print(f"Rouge-P: {np.mean(m_rouge_p)}")
    print(f"Cos-Sim: {np.mean(m_cos_sim)}")

    logger.log({f"{prefix}/avg/nll": np.mean(m_nll),
                f"{prefix}/avg/ppl": np.mean(m_ppl),
                f"{prefix}/avg/levenstein": np.mean(m_levenstein),
                f"{prefix}/avg/rouge-r": np.mean(m_rouge_r),
                f"{prefix}/avg/rouge-f": np.mean(m_rouge_f),
                f"{prefix}/avg/rouge-p": np.mean(m_rouge_p),
                f"{prefix}/avg/cos-sim": np.mean(m_cos_sim),
                })



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
    bert_model.load_state_dict(torch.load(artifact_dir + '/code-bert.pth', map_location=torch.device(config.device)))
    bert_model = bert_model.to(config.device)

    # load eval dataset
    config.eos_token_id = tokenizer.encode("<EOL>")[0]
    config.pad_token_id = tokenizer.encode("<pad>")[0]
    eval_data = tokenize_files(config.validation_data, tokenizer, config)
    eval = TextDataset(inp=eval_data, block_size=config.block_size, eos_token_id=config.eos_token_id,
                       pad_token_id=config.pad_token_id)

    for evaluation in models:
        run_evaluation(logger, config, evaluation, tokenizer, eval, bert_model)
