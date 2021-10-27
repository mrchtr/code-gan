import torch
import wandb
from datasets import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, DataCollatorWithPadding, DataCollatorForWholeWordMask, RobertaForMaskedLM, AdamW, \
    AutoModelForCausalLM

from config import init_config
from data.Dataset import TextDataset


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

if __name__ == '__main__':
    epochs = 10
    print("Start fine-tuning BERT model ...")

    # init wandb & config
    config = init_config()
    logger = wandb.init(project=config.project_name, config=config)


    # using gpt2 tokenizer
    tokenizer = GPT2Tokenizer(vocab_file="code-tokenizer-vocab.json", merges_file="code-tokenizer-merges.txt")
    tokenizer.add_tokens(config.special_tokens)
    config.eos_token_id = tokenizer.encode("<EOL>")[0]
    config.pad_token_id = tokenizer.encode("<pad>")[0]
    config.vocab_size = len(tokenizer)

    # load mask bert model
    model = AutoModelForCausalLM.from_pretrained("huggingface/CodeBERTa-small-v1")
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(config.device)

    training_data = tokenize_files(config.training_data, tokenizer, config)
    train = TextDataset(inp=training_data, block_size=config.block_size, eos_token_id=config.eos_token_id,
                        pad_token_id=config.pad_token_id)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )


    print("INFO: Start huggingface pretraining ... ")

    train_dataloader = DataLoader(train, shuffle=True, batch_size=64)


    optimizer = Adam(model.parameters(), lr=4e-4)

    progress_bar = tqdm(range(epochs * train.__len__()))

    model.train()
    for epoch in range(epochs):
        i = 0
        for batch in train_dataloader:
            input = batch[0].to(config.device)
            outputs = model(input_ids=input, labels=input)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            logger.log({f"bert_pretrain/loss": loss.item()})


            if i % 10000 == 0:
                torch.save(model.state_dict(), 'code-bert.pth')
                artifact = wandb.Artifact('codeberta', type='model')
                artifact.add_file('code-bert.pth')
                logger.log_artifact(artifact)
                i+=1