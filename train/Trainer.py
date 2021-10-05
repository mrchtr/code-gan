import math

import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as f

import numpy as np
from torchtext.data.metrics import bleu_score

from models.discriminator.Discriminator import Discriminator
from models.generator.Generator import Generator
from train.Metrics import Metrics
from utils.FileUtils import create_dir_if_not_exists
from tqdm import tqdm

from utils.Metrics import get_bleu

text_table = wandb.Table(columns=["sample"])


class Trainer:
    """
    Holding both models for the adversarial training and the main training loop.
    """

    def __init__(self, generator, discriminator, dataset, tokenizer, config, logger=None, reference_corpus=None):
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
        self.lr_adv_g = config.lr_adv_g
        self.lr_adv_d = config.lr_adv_d
        self.lr_pretrain = config.lr_pretrain
        self.nadv_steps = config.nadv_steps
        self.g_steps = config.g_steps
        self.d_steps = config.d_steps

        self.pretrain_optimizer = config.pretrain_optimizer
        self.generator_optimizer = config.generator_optimizer
        self.discriminator_optimizer = config.discriminator_optimizer

        self.dataset = dataset
        self.dataloader = DataLoader(dataset, self.batch_size, drop_last=True, shuffle=True)
        # self.dataloader_eval = DataLoader(reference_corpus, self.batch_size, drop_last=True, shuffle=True)
        self.tokenizer = tokenizer
        self.test_file = config.validation_data

        self.device = config.device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.metrics = Metrics(config.sequence_length)
        self.logger = logger

    @staticmethod
    def _get_optimizer(name, parameters, lr):
        if name == "SGD":
            return optim.SGD(parameters, lr=lr)
        elif name == "Adam":
            return optim.Adam(parameters, lr=lr)
        elif name == "AdamW":
            return optim.AdamW(parameters, lr=lr)
        else:
            raise Exception(f"Can't create unknown optimizer {name}")

    def train(self):
        """
        Main training loop. Including pretraining and adverserial training
        """
        # pretrained model perplexity
        self.generate_sample()
        self._pretrain_generator()
        self._adversarial_training()

        # final evaluation
        self.eval_generator()

    def generate_sample(self):
        print("Testing sample generation of random initialized generator: ")
        try:
            context, ground_truth = self._generate_context()
            sample = self.generator.sample(context, self.sequence_length, self.batch_size, num_samples=1).to(
                'cpu')  # array of sample tokens

            sample_str = self.tokenizer.decode(sample.numpy()[0].tolist())
            print(f"Given:        {self.tokenizer.decode(context[0].numpy())}")
            print(f"Proposed:     {sample_str}")
            print(f"Ground Truth: {self.tokenizer.decode(ground_truth[0].numpy())}")
            text_table.add_row(sample_str)
            self.logger.log({"samples": text_table})
        except:
            print(f"Error while generating sample")

    def _pretrain_generator(self):
        print("Start pretraining of generator ...")

        if self.config.generator == "GPTCode":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.NLLLoss()  # softmax already applied inside the model
        optimizer = self._get_optimizer(self.pretrain_optimizer, self.generator.parameters(), lr=self.lr_pretrain)
        losses = []
        iterator = iter(self.dataloader)

        # initial hidden state
        hidden = self.generator.init_state(self.batch_size)
        self.generator.train()

        for i in tqdm(range(self.config.pretraining_steps)):
            x, y = next(iterator)
            x = x.to(self.device)
            y = y.to(self.device)

            pred, hidden, next_token = self.generator(x, hidden)

            shift_logits = pred[..., :-1, :].contiguous()  # remove the last logits in every batch
            shift_labels = y[..., 1:].contiguous()  # removing the first tokens in each label sequence
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            self.logger.log({f"pretraining/loss": loss.item()})

        print(f"Mean losses: {np.mean(losses)}")
        torch.save(self.generator.state_dict(), 'generator.pth')

    def _adversarial_training(self):
        print(f"Start adversarial training. Run for {self.nadv_steps} steps ...")
        generator_optimizer = self._get_optimizer(self.generator_optimizer, self.generator.parameters(),
                                                  lr=self.lr_adv_g)
        discriminator_optimizer = self._get_optimizer(self.discriminator_optimizer, self.discriminator.parameters(),
                                                      lr=self.lr_adv_d)

        for i in tqdm(range(self.nadv_steps)):
            # context should be of shape (batch_size, block_size)
            x, real_data = self._generate_context()

            loss_g = self.adv_train_generator(x, real_data, generator_optimizer)
            loss_d = self.adv_train_discriminator(x, real_data, discriminator_optimizer)
            print(f"D_Loss: {loss_d.item()} - G_Loss: {loss_g.item()}")

            self.eval_generator()
            # update temperature each epoch
            self.generator.temperature = self.update_temperature(self.generator.temperature, i)

            self.logger.log({"generator/loss": loss_g, "discriminator/loss": loss_d,
                             "temperature": self.generator.temperature})

            if i % 100 == 0:
                torch.save(self.generator.state_dict(), 'generator.pth')


    def _generate_context(self):
        if self.config.noise_as_context:
            return torch.LongTensor([0] * self.batch_size * self.config.block_size).reshape(self.batch_size,
                                                                                            self.config.block_size).to(
                self.device)
        else:
            context, ground_truth = self.dataset.get_random_context_with_ground_truth(self.batch_size,
                                                                                      self.config.start_sequence_len,
                                                                                      self.config.sequence_length)
            return context.to(self.device), ground_truth.to(self.device)

    def evaluate_generator(self, epoch):
        # calculate perplexit
        # for i in range(2, 6):
        #    bleu = bleu_score(sample, self.reference_corpus, max_n=i)
        #    self.logger.log({f"bleu-{i}": bleu})

        # self.eval_generator()

        # ---- generate data
        try:
            x = self._generate_context()
            sample = self.generator.sample(x, self.sequence_length, self.batch_size, num_samples=1).to(
                'cpu')  # array of sample tokens

            sample_str = self.tokenizer.decode(sample.numpy()[0].tolist())
            # print(f"Sample: {sample_str}")
            # ---- logging to wandb
            text_table.add_data(epoch, sample_str)
            self.logger.log({"samples": text_table})
        except:
            print(f"Error while evaluation")

    def adv_train_generator(self, x, real_data, optimizer):
        losses = []
        for i in range(self.g_steps):
            generated_data = self.generator.sample(x, self.sequence_length, self.batch_size,
                                                   num_samples=self.batch_size).to(self.device)

            discriminator_real_out = self.discriminator(self.prepare_d_inp(real_data))
            discriminator_fake_out = self.discriminator(self.prepare_d_inp(generated_data))

            loss_g, _ = self.get_losses(discriminator_real_out, discriminator_fake_out)

            self.generator.zero_grad()
            loss_g.backward(retain_graph=False)
            optimizer.step()
            losses.append(loss_g.item())

        return np.mean(losses)

    def adv_train_discriminator(self, x, real_data, optimizer):
        losses = []
        for i in range(self.d_steps):
            generated_data = self.generator.sample(x, self.sequence_length, self.batch_size,
                                                   num_samples=self.batch_size).to(self.device)

            discriminator_real_out = self.discriminator(self.prepare_d_inp(real_data))
            discriminator_fake_out = self.discriminator(self.prepare_d_inp(generated_data))
            gradient_penalty = self.calculate_gradient_penalty(real_data, generated_data)
            _, loss_d = self.get_losses(discriminator_real_out, discriminator_fake_out, gradient_penalty)

            self.discriminator.zero_grad()
            loss_d.backward(retain_graph=False)
            optimizer.step()
            losses.append(loss_d.item())

        return np.mean(losses)

    def eval_generator(self):
        self.generator.eval()
        with torch.no_grad():

            # get context
            x, real_data = self._generate_context()

            # create sample
            generated_data = self.generator.sample(x, self.sequence_length, self.batch_size,
                                                   num_samples=self.batch_size).to(self.device)
            # get string represenation
            generated_data = self.tokenizer.decode_batch(generated_data.numpy(), skip_special_tokens=False)
            real_data = self.tokenizer.decode_batch(real_data.numpy(), skip_special_tokens=False)
            bleu = []
            nll = []
            perplexity = []
            edit_sm = []

            # bleu
            for generated, real in generated_data, real_data:
                bleu.append(get_bleu(generated, real))

            # generator metrics:
            #   - BLEU score
            #   - NLL
            #   - Perplexity
            #   - Edit Similarity


            # discriminator metrics
            #   - accuracy

            sample = self.dataset.get_random_real_sample(1, self.sequence_length)
            x = sample[:, 0:self.config.start_sequence_len]
            x = x.to(self.device)
            reference = self.tokenizer.decode(sample[0].numpy(), skip_special_tokens=False)
            output = self.generator.sample(x, self.sequence_length, 1)
            predicition = self.tokenizer.decode(sample[0].numpy(), skip_special_tokens=False)
            euclidian, cos_sim, levenstein, bleu = self.metrics.get_similarity(predicition, reference)
            self.logger.log(
                {'euclidian': euclidian, 'cosisinus_sim': cos_sim, 'levenstein_dis': levenstein, 'blue': bleu})

        self.generator.train()

    def prepare_d_inp(self, inp):
        return f.one_hot(inp, self.config.vocab_size).float()

    def get_losses(self, d_out_real, d_out_fake, gradient_penalty=0):
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
        if self.config.loss_type == "standard":  # wasserstein loss
            d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real)).to(self.device)
            d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake)).to(self.device)
            d_loss = d_loss_real + d_loss_fake

            g_loss = bce_loss(d_out_fake, torch.ones_like(d_out_fake))

        elif "wgan" in self.config.loss_type:  # 'wgan' or 'wgan-gp'
            """
            see for definition: https://arxiv.org/abs/1704.00028v3
            basically: 
            errD = -errD_real + errD_fake + gradient_penalty * 10
            errG = -mean(D(fake)) = -torch.mean(dis_fake)
            """

            d_loss_real = d_out_real.mean()
            d_loss_fake = d_out_fake.mean()
            d_loss = -d_loss_real + d_loss_fake
            g_loss = -d_loss_fake

            if 'gp' in self.config.loss_type:
                d_loss = d_loss + gradient_penalty

        else:  # relativistic standard GAN (rsgan)
            d_loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real)).to(self.device)
            g_loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake)).to(self.device)

        return d_loss, g_loss

    def update_temperature(self, temperature, i):
        """
        Updating temperature of generator. Same methode like it is done in amazon Transformer GAN for music generation
        :param temperature: current temperature
        :param current_epoch: current epoch of training
        :param max_epoch: max epochs of training
        :return: temperature
        """
        beta_max = self.config.temperature
        N = self.nadv_steps
        return beta_max ** (i / N)

    def get_metrics(self, gen_file, test_file):
        for i in range(2, 6):
            bleu = Bleu(test_text=gen_file, real_text=test_file, gram=i, name=f"blue-{i}")
            self.logger.log({f"bleu-{i}": bleu.get_bleu()})

    def calculate_gradient_penalty(self, real_data, fake_data, LAMBDA=10):
        """
        Calculates the gradient penalty loss for WGAN GP
        source: https://github.com/Lornatang/WassersteinGAN_GP-PyTorch/
        """
        # Random weight term for interpolation between real and fake data
        real_data = self.prepare_d_inp(real_data)
        fake_data = self.prepare_d_inp(fake_data)
        alpha = torch.rand([real_data.shape[0], 1, 1], device=self.device)
        alpha = alpha.expand(real_data.size())

        # Get random interpolation between real and fake data
        # a = 0.1 r = 6580 f = 9456
        # 0.1 * 6580 + ((1 - 0.1) * 9456)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        dis_interpolates = self.discriminator(interpolates)

        grad_outputs = torch.ones(dis_interpolates.size(), device=self.device, requires_grad=False)
        gradients = torch.autograd.grad(outputs=dis_interpolates, inputs=interpolates,
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(real_data.shape[0], -1)

        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((slopes - 1.) ** 2).mean() * LAMBDA

        return gradient_penalty
