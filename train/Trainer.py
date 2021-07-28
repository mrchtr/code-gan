import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader

import numpy as np

from models.discriminator.Discriminator import Discriminator
from models.generator.Generator import Generator
from utils.FileUtils import create_dir_if_not_exists
from utils.Bleu import Bleu
from tqdm import tqdm
text_table = wandb.Table(columns=["epoch", "sample"])

class Trainer:
    """
    Holding both models for the adversarial training and the main training loop.
    """
    def __init__(self, generator, discriminator, dataset, tokenizer, config, logger=None):
        """
        :param generator: Generator model
        :param discriminator: Discriminator model
        :param sequence_length: sequence length
        :param batch_size: batch size
        :param lr: learning rate
        :param max_epochs: maximum of epochs to train
        """
        self.config = config
        self.generator: Generator = generator
        self.discriminator: Discriminator = discriminator
        self.sequence_length = config.sequence_length
        self.batch_size = config.batch_size
        self.lr_adv = config.lr_adv
        self.lr_pretrain = config.lr_pretrain
        self.nadv_steps = config.nadv_steps
        self.g_steps = config.g_steps
        self.d_steps = config.d_steps

        self.pretrain_optimizer = config.pretrain_optimizer
        self.generator_optimizer = config.generator_optimizer
        self.discriminator_optimizer = config.discriminator_optimizer

        self.dataset = dataset
        self.dataloader = DataLoader(dataset, self.batch_size, drop_last=True)
        self.tokenizer = tokenizer
        self.test_file = config.validation_data

        self.device = config.device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.logger = logger

    def train(self):
        """
        Main training loop. Including pretraining and adverserial training
        """

        self._pretrain_generator()
        self._adversarial_training()

        # logging samples to wandb
        self.logger.log({"generated samples": text_table})

    @staticmethod
    def _get_optimizer(name, parameters, lr):
        if name == "SGD":
            return optim.SGD(parameters, lr=lr)
        elif name == "Adam":
            return optim.Adam(parameters, lr=lr)
        else:
            raise Exception(f"Can't create unknown optimizer {name}")

    def _adversarial_training(self):
        print(f"Start adversarial training. Run for {self.nadv_steps} steps ...")
        generator_optimizer = self._get_optimizer(self.generator_optimizer, self.generator.parameters(), lr=self.lr_adv)
        discriminator_optimizer = self._get_optimizer(self.discriminator_optimizer, self.discriminator.parameters(), lr=self.lr_adv)

        for i in tqdm(range(self.nadv_steps)):
            x = torch.LongTensor([0] * self.batch_size * self.config.block_size).reshape(self.batch_size, self.config.block_size).to(self.device)

            loss_g = self.adv_train_generator(x, generator_optimizer)
            loss_d = self.adv_train_discriminator(x, discriminator_optimizer)

            # update temperature each epoch
            self.generator.temperature = self.update_temperature(self.generator.temperature, i)
            self.evaluate_generator(i)
            self.logger.log({"generator/loss": loss_g, "discriminator/loss": loss_d,
                             "temperature": self.generator.temperature})

            if self.nadv_steps % 20 == 0:
                torch.save(self.generator.state_dict(), 'generator.pth')


    def evaluate_generator(self, epoch):
        create_dir_if_not_exists("sample_dir")
        sample_dir = "sample_dir/generated_sample.txt"
        # ---- generate data
        x = torch.LongTensor([0] * self.batch_size * self.config.block_size).reshape(self.batch_size, self.config.block_size).to(self.device)
        sample = self.generator.sample(x, self.sequence_length, self.batch_size, num_samples=1).to('cpu')
        sample_str = self.tokenizer.decode(sample.numpy()[0].tolist())

        with open(sample_dir, 'w') as outfile:
            outfile.write(sample_str)
        outfile.close()
        # ---- logging to wandb
        text_table.add_data(epoch, sample_str)


        # ---- calculate bleu score
        return self.get_metrics(sample_dir, self.test_file)


    def adv_train_generator(self, x, optimizer):
        losses = []
        for i in range(self.g_steps):
            real_data = self.dataset.get_random_real_sample(self.batch_size, self.sequence_length).to(
                self.device)
            generated_data = self.generator.sample(x, self.sequence_length, self.batch_size, num_samples=self.batch_size).to(self.device)

            discriminator_real_out = self.discriminator(real_data)
            discriminator_fake_out = self.discriminator(generated_data)

            loss_g, _ = self.get_losses(discriminator_real_out, discriminator_fake_out)

            self.generator.zero_grad()
            loss_g.backward(retain_graph=False)
            optimizer.step()
            losses.append(loss_g.item())

        return np.mean(losses)

    def adv_train_discriminator(self, x, optimizer):
        losses = []
        for i in range(self.d_steps):
            real_data = self.dataset.get_random_real_sample(self.batch_size, self.sequence_length).to(
                self.device)
            generated_data = self.generator.sample(x, self.sequence_length, self.batch_size, num_samples=self.batch_size).to(self.device)

            discriminator_real_out = self.discriminator(real_data)
            discriminator_fake_out = self.discriminator(generated_data)

            _, loss_d = self.get_losses(discriminator_real_out, discriminator_fake_out)

            self.discriminator.zero_grad()
            loss_d.backward(retain_graph=False)
            optimizer.step()
            losses.append(loss_d.item())

        return np.mean(losses)

    def _pretrain_generator(self):
        print("Start pretraining of generator ...")
        criterion = nn.NLLLoss()  # softmax already applied inside the model
        optimizer = self._get_optimizer(self.pretrain_optimizer, self.generator.parameters(), lr=self.lr_pretrain)
        losses = []
        loss_per_epoch = []
        for epoch in range(self.config.pretraining_epochs):
            print(f"Run pretrain epoch {epoch} ...")
            hidden = self.generator.init_state(self.batch_size)
            self.generator.train()
            for batch, (x, y) in enumerate(tqdm(self.dataloader)):
                x = x.to(self.device)
                y = y.to(self.device)

                if self.config.generator == "Transformer":
                    hidden = hidden.to(self.device)
                else:
                    hidden = hidden[0].to(self.device), hidden[1].to(self.device)

                # if y contains a whole sequence just using the last token
                if y.shape[1] > 1:
                    y = y[:, -1]

                pred, hidden, next_token = self.generator(x, hidden)
                loss = criterion(pred, y.view(-1))

                if self.config.generator == "LSTM":
                    hidden = hidden[0].detach(), hidden[1].detach()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                #self.logger.log({f"{epoch}/loss": loss.item()})
            self.logger.log({f"pretraining/loss": np.mean(losses)})

        torch.save(self.generator.state_dict(), 'generator.pth')


    def get_losses(self, d_out_real, d_out_fake):
        """
        Calculates the losses based on d_out_real and d_out_fake
        Discriminator: max log(D(real)) + log ( 1 - D(G(z))
        Generator: min log (1 - D(G(z)) <-> max log(D(G(z))
        :param d_out_real:
        :param d_out_fake:
        :return: d_loss, g_loss
            - d_loss: discirminator loss value
            - g_loss: generator loss value
        """
        bce_loss = nn.BCEWithLogitsLoss()

        d_loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real)).to(self.device)
        g_loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake)).to(self.device)

        return d_loss, g_loss

    def update_temperature(self, temperature, i):
        """
        Updating temperature of generator. For now just linear decrease.
        :param temperature: current temperature
        :param current_epoch: current epoch of training
        :param max_epoch: max epochs of training
        :return: temperature
        """
        N = self.nadv_steps  # todo implement real method
        return 1 + i / (N - 1) * (temperature - 1)


    def get_metrics(self, gen_file, test_file):
        for i in range(2, 6):
            bleu = Bleu(test_text=gen_file, real_text=test_file, gram=i, name=f"blue-{i}")
            self.logger.log({f"bleu-{i}": bleu.get_bleu()})
