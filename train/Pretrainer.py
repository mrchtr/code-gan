from torch import optim
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, TextDataset, GPT2Config, \
    GPT2LMHeadModel


class Pretrainer:
    """
    Pretraining GPT2 and Bert on source code dataset.
    """

    def __init__(self, model, tokenizer, config):
        if model == "GPT2":
            configuration = GPT2Config(
                vocab_size=config.vocab_size
            )
            self.model = GPT2LMHeadModel(configuration)

        self.config = config
        self.tokenizer = tokenizer

    def train(self):
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=self.config.training_data,
            block_size=128)

        test_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=self.config.validation_data,
            block_size=128)


        data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False,
            )

        training_args = TrainingArguments(
            output_dir="./gpt2-code-pretrained",  # The output directory
            overwrite_output_dir=True,  # overwrite the content of the output directory
            num_train_epochs=3,  # number of training epochs
            per_device_train_batch_size=32,  # batch size for training
            per_device_eval_batch_size=64,  # batch size for evaluation
            eval_steps=400,  # Number of update steps between two evaluations.
            save_steps=800,  # after # steps model is saved
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            prediction_loss_only=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()
        trainer.save_model("./gpt2-code-pretrained")

