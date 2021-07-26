import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data.Dataset import CodeDataset
from models.Discriminator import Discriminator
from models.Generator import Generator
import numpy as np

from utils.FileUtils import create_dir_if_not_exists
from utils.Bleu import Bleu


class Trainer:
    """
    Holding both models for the adversarial training and the main training loop.
    """
    def __init__(self, generator, discriminator, sequence_length, dataset, batch_size=16, lr_pretrain=1e-2, lr_adv=1e-4, max_epochs=10,
                 nadv_steps=2000, g_steps=1, d_steps=5, tokenizer=None, test_file=None):
        """
        :param generator: Generator model
        :param discriminator: Discriminator model
        :param sequence_length: sequence length
        :param batch_size: batch size
        :param lr: learning rate
        :param max_epochs: maximum of epochs to train
        """
        self.generator: Generator = generator
        self.discriminator: Discriminator = discriminator
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.lr_adv = lr_adv
        self.lr_pretrain = lr_pretrain
        self.max_epochs = max_epochs
        self.nadv_steps = nadv_steps
        self.g_steps = g_steps
        self.d_steps = d_steps
        self.dataset: CodeDataset = dataset
        self.dataloader = DataLoader(dataset, batch_size, drop_last=True)
        self.tokenizer = tokenizer
        self.test_file = test_file

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)


    def train(self, pretraining_generator=True, pretrain_epochs=1):
        """
        Main training loop. Including pretraining and adverserial training
        """
        if pretraining_generator:
            pretraining_losses = self._pretrain_generator(pretrain_epochs)

        adversarial_losses_gen, adversarial_losses_dis, metrics_summary = self._adversarial_training()

        return pretraining_losses, adversarial_losses_gen, adversarial_losses_dis, metrics_summary

    def _adversarial_training(self):
        print("Start adversarial training ... ")
        generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr_adv)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr_adv)
        losses_per_epoch_generator = []
        losses_per_epoch_discriminator = []

        g_losses = []
        d_losses = []
        metrics_summary= []
        for i in range(self.nadv_steps):
            # x = x.to(self.device) todo: later decide which context to use
            x = torch.LongTensor([0] * self.batch_size).reshape(self.batch_size, 1).to(self.device)
            loss_g = self.adv_train_generator(x, generator_optimizer)
            loss_d = self.adv_train_discriminator(x, discriminator_optimizer)

            g_losses.append(loss_g)
            d_losses.append(loss_d)

            # update temperature each epoch
            self.generator.temperature = self.update_temperature(self.generator.temperature, i)

            metrics = self.evaluate_generator()
            metrics_summary.append(metrics)

            if i % 10 == 0:
                print(f"Epoch: {i}, l_g: {np.mean(g_losses)}, l_d: {np.mean(d_losses)}, temperature: {self.generator.temperature}")
                print(f"Bleu score: {metrics[0]:.3f} / {metrics[1]:.3f} / {metrics[2]:.3f} / {metrics[3]:.3f}")

            losses_per_epoch_generator.append(np.mean(g_losses))
            losses_per_epoch_discriminator.append(np.mean(d_losses))

        return losses_per_epoch_generator, losses_per_epoch_discriminator, metrics_summary

    def evaluate_generator(self):
        create_dir_if_not_exists("sample_dir")
        sample_dir = "sample_dir/generated_sample.txt"
        # ---- generate data
        x = torch.LongTensor([0] * self.batch_size).reshape(self.batch_size, 1).to(self.device)
        sample = self.generator.sample(x, self.sequence_length, self.batch_size, num_samples=1).to('cpu')
        sample_str = self.tokenizer.decode(sample.numpy()[0].tolist())

        with open(sample_dir, 'w') as outfile:
            outfile.write(sample_str)
        outfile.close()

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

    def _pretrain_generator(self, epochs):
        print("Start pretraining of generator ...")
        criterion = nn.NLLLoss()  # softmax already applied inside the model
        optimizer = optim.Adam(self.generator.parameters(), lr=self.lr_pretrain)
        losses = []
        loss_per_epoch = []
        for epoch in range(epochs):
            hidden = self.generator.init_state(self.batch_size)
            self.generator.train()

            for batch, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                hidden = hidden[0].to(self.device), hidden[1].to(self.device)

                pred, hidden, next_token = self.generator(x, hidden)
                loss = criterion(pred, y.view(-1))
                hidden = hidden[0].detach(), hidden[1].detach()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch} - loss: {np.mean(losses)}")

            loss_per_epoch.append(np.mean(losses))

        return loss_per_epoch


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
        N = 5000  # todo implement real method
        return 1 + i / (N - 1) * (temperature - 1)


    def get_metrics(self, gen_file, test_file):
        metrics = []
        for i in range(2, 6):
            bleu = Bleu(test_text=gen_file, real_text=test_file, gram=i, name=f"blue-{i}")
            metrics.append(bleu.get_bleu())
        return metrics
