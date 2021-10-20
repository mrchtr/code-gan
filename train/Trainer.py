import math

import jellyfish
import torch
import wandb
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.nn import CrossEntropyLoss
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

    def __init__(self, generator, discriminator, dataset, tokenizer, config, logger=None, dataset_eval = None):
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
        self.dataset_eval = dataset_eval
        self.dataloader = DataLoader(dataset, self.batch_size, drop_last=True, shuffle=True)
        # self.dataloader_eval = DataLoader(reference_corpus, self.batch_size, drop_last=True, shuffle=True)
        self.tokenizer = tokenizer
        self.test_file = config.validation_data

        self.device = config.device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        #self.metrics = Metrics(config.sequence_length)
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
        global global_log_step
        # pretrained model perplexity
        self.generate_sample()

        print("Validate generator initial state: ")
        self.eval_generator()

        self._pretrain_generator()

        self._adversarial_training()

        # final evaluation
        self.eval_generator(validation_set=True)

    def generate_sample(self):
        print(60 * "-")
        print("Generates Test Sample")
        try:
            context, ground_truth = self._generate_context()
            sample = self.generator.sample(context, self.sequence_length, self.batch_size, num_samples=1).to(
                'cpu')  # array of sample tokens
            sample = sample[:, self.config.start_sequence_len:self.config.sequence_length]
            sample_str = self.tokenizer.decode(sample.numpy()[0].tolist(), skip_special_tokens=False)
            print(f"Given:        {self.tokenizer.decode(context[0].to('cpu').numpy(), skip_special_tokens=False)}")
            print(f"Proposed:     {sample_str}")
            print(f"Ground Truth: {self.tokenizer.decode(ground_truth[0].to('cpu').numpy(), skip_special_tokens=False)}")
            print(60 * "-")
        except:
            print(f"Error while generating sample")

    def _pretrain_generator(self):
        print("Start pretraining of generator ...")

        if self.config.generator == "GPTCode":
            criterion = nn.CrossEntropyLoss()
            #criterion = nn.NLLLoss()  # softmax already applied inside the model
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

            logits = self.generator(x, hidden, return_dict=True).logits

            shift_logits = logits[..., :-1, :].contiguous()  # remove the last logits in every batch
            shift_labels = y[..., 1:].contiguous()  # removing the first tokens in each label sequence
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.clip_norm)
            optimizer.step()
            losses.append(loss.item())
            #print(loss.item())
            self.logger.log({f"pretraining/loss": loss.item(), "iteration" : i})

            if i % 100 == 0:
                self.generate_sample()
                torch.save(self.generator.state_dict(), 'generator.pth')

        print(f"Mean losses: {np.mean(losses)}")

    def _adversarial_training(self):
        print(f"Start adversarial training. Run for {self.nadv_steps} steps ...")
        generator_optimizer = self._get_optimizer(self.generator_optimizer, self.generator.parameters(),
                                                  lr=self.lr_adv_g)
        discriminator_optimizer = self._get_optimizer(self.discriminator_optimizer, self.discriminator.parameters(),
                                                      lr=self.lr_adv_d)

        for i in tqdm(range(self.nadv_steps)):
            # context should be of shape (batch_size, block_size)
            loss_g = self.adv_train_generator(generator_optimizer)
            loss_d = self.adv_train_discriminator(discriminator_optimizer)
            print(f"D_Loss: {loss_d.item()} - G_Loss: {loss_g.item()}")

            bleu, es, ppl  = self.eval_generator()
            # update temperature each epoch
            self.generator.temperature = self.update_temperature(self.generator.temperature, i)

            self.logger.log({"generator/loss": loss_g, "discriminator/loss": loss_d,
                             "temperature": self.generator.temperature, "generator/bleu": bleu,
                             "generator/edit_similarity": es,
                             "generator/ppl": ppl,
                             "iteration": i})

            if i % 100 == 0:
                torch.save(self.generator.state_dict(), 'generator.pth')
                artifact = wandb.Artifact('model', type='model')
                artifact.add_file('generator.pth')
                self.logger.log_artifact(artifact)
                self.generate_sample()


    def _generate_context(self, validation_set = False):
        if validation_set:
            dataset = self.dataset_eval
        else:
            dataset = self.dataset
        if self.config.noise_as_context:
            return torch.LongTensor([0] * self.batch_size * self.config.block_size).reshape(self.batch_size,
                                                                                            self.config.block_size).to(
                self.device)
        else:
            context, ground_truth = dataset.get_random_context_with_ground_truth(self.batch_size,
                                                                                      self.config.start_sequence_len,
                                                                                      self.config.sequence_length)
            return context.to(self.device), ground_truth.to(self.device)

    # todo check if we can combine train dis and gen to one method
    def adv_train_generator(self, optimizer):
        x, real_data = self._generate_context()
        losses = []
        for i in range(self.g_steps):
            generated_data = self.generator.sample(x, self.sequence_length, self.batch_size,
                                                   num_samples=self.batch_size).to(self.device)
            #generated_data = generated_data[:, self.config.start_sequence_len:self.config.sequence_length]
            discriminator_real_out = self.discriminator(self.prepare_d_inp(real_data))
            discriminator_fake_out = self.discriminator(self.prepare_d_inp(generated_data))

            loss_g, _ = self.get_losses(discriminator_real_out, discriminator_fake_out)

            optimizer.zero_grad()
            loss_g.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.clip_norm)
            optimizer.step()
            losses.append(loss_g.item())

        return np.mean(losses)

    def adv_train_discriminator(self, optimizer):
        x, real_data = self._generate_context()
        losses = []
        for i in range(self.d_steps):
            generated_data = self.generator.sample(x, self.sequence_length, self.batch_size,
                                                   num_samples=self.batch_size).to(self.device)
            #generated_data = generated_data[:, self.config.start_sequence_len:self.config.sequence_length]

            discriminator_real_out = self.discriminator(self.prepare_d_inp(real_data))
            discriminator_fake_out = self.discriminator(self.prepare_d_inp(generated_data))
            gradient_penalty = self.calculate_gradient_penalty(real_data, generated_data)
            _, loss_d = self.get_losses(discriminator_real_out, discriminator_fake_out, gradient_penalty)

            optimizer.zero_grad()
            loss_d.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.clip_norm)
            optimizer.step()
            losses.append(loss_d.item())

        return np.mean(losses)

    def eval_generator(self, validation_set=False):
        """
        Provide following metrics:
        - BLEU Score : how could the train set was learned
        - Classifier Accuracy : lower accuracy, better generated results
        - Edit Similarity
        - Perplexity
        :return:
        """
        self.generator.eval()
        self.discriminator.eval()

        bleu = []
        levenstein = []

        with torch.no_grad():

            # get context
            context_token, real_data_token = self._generate_context(validation_set=True)

            # create sample
            generated_data_token = self.generator.sample(context_token, self.sequence_length, self.batch_size,
                                                   num_samples=self.batch_size).to(self.device)
            # get string represenation
            generated_data_str = self.tokenizer.batch_decode(generated_data_token.to('cpu').numpy(), skip_special_tokens=False)
            real_data_str = self.tokenizer.batch_decode(real_data_token.to('cpu').numpy(), skip_special_tokens=False)

            # bleu & levenstein
            for i in range(0, len(generated_data_str)):
                generated = generated_data_str[i]
                real = real_data_str[i]
                bleu.append(get_bleu(generated, real))
                levenstein.append(jellyfish.levenshtein_distance(generated, real))

            #perplexity
            lm_logits = self.generator(context_token, return_dict=True).logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = context_token[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ppl = torch.exp(loss.mean())
            # classifier accuracy
            #dis_inp = torch.cat((real_data_token, generated_data_token))
            #dis_y = [1] * real_data_token.shape[0] + [0] * generated_data_token.shape[0]  # real-label: 1, fake-lable: 0
            #y_pred = self.discriminator(self.prepare_d_inp(dis_inp))
            #y_pred = torch.round(torch.sigmoid(y_pred)).to('cpu').numpy()
            #accuracy = accuracy_score(dis_y, y_pred, normalize=True)


        self.generator.train()
        self.discriminator.train()
        return np.mean(bleu), np.mean(levenstein), ppl

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
