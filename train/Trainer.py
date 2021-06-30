import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data.CodeDataset import CodeDataset
from models.Discriminator import Discriminator
from models.Generator import Generator


class Trainer:
    """
    Holding both models for the adversarial training and the main training loop.
    """
    def __init__(self, generator, discriminator, sequence_length, dataset, batch_size=16, lr=1e-4, max_epochs=10):
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
        self.lr = lr
        self.max_epochs = max_epochs
        self.dataset: CodeDataset = dataset
        self.dataloader = DataLoader(dataset, batch_size, drop_last=True)

    def train(self):
        generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        discriminator_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)

        for epoch in range(self.max_epochs):

            for batch, (x, y) in enumerate(self.dataloader):
                generated_data = self.generator.sample(x, self.sequence_length, self.batch_size)
                real_data = self.dataset.get_random_real_sample(self.batch_size)

                self.discriminator.zero_grad()
                self.generator.zero_grad()

                discriminator_real_out = self.discriminator(real_data)
                discriminator_fake_out = self.discriminator(generated_data)
                loss_g, loss_d = self.get_losses(discriminator_real_out, discriminator_fake_out)

                loss_d.backward(retain_graph=True)
                discriminator_optimizer.step()
                loss_g.backward(retain_graph=True)
                generator_optimizer.step()

                #print(f"Train step: {batch}, loss generator: {loss_g}, loss discriminator: {loss_d}")

            self.generator.temperature = self.update_temperature(self.generator.temperature, epoch, self.max_epochs)
            tokenizer = self.dataset.tokenizer
            if epoch % 1 == 0:
                print(f"Train step: {batch}, loss generator: {loss_g}, loss discriminator: {loss_d}")
                print(f"Example: {tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(generated_data[0]))}")

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
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake
        g_loss = -d_loss_fake

        return d_loss, g_loss

    def update_temperature(self, temperature, current_epoch, max_epoch):
        """
        Updating temperature of generator. For now just linear decrease.
        :param temperature: current temperature
        :param current_epoch: current epoch of training
        :param max_epoch: max epochs of training
        :return: temperature
        """
        return temperature - (current_epoch / max_epoch)