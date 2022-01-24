# Generative adversarial network for source code suggestions

Code for the masterthesis 'Generative adversarial network for source code suggestions'

If you use this code please cite the paper using the following bib tex:

```
@mastersthesis{code-gan,
  title   = {Generative adversarial network for source code suggestions},
  author  = {Matthias Richter},
  school  = {Universität Leipzig},
  year    = {2021}
}
```

## Requirements
- Python 3.6+
- Pip

For installing all dependencies please use:
```
pip install requirements.txt
```

## Dataset

For the training the [150k Python Dataset](https://eth-sri.github.io/py150) was used.
The complete dataset can be downloaded by: 

```
python3 get_data.py
```

For each file in the dataset the preprocessing steps will be applied. 
Afterwards, the whole dataset will be tokenized.

## Train Generator

To run the generator train change the configuration file and run the following: 
```
python3 train.py
```

Pseudo Code:
```py
# Init Generator with random weights
G = Generator()

# Init Discrimintaor
D = Discriminator()

# Pretrain Generator
for x, y in train:
    generator(x)
    loss = criterion(x,y)
    loss.step

# Adversarial Training
for n in steps:
    
    
    # train generator
    condition, ground_truth = dataset.get_sample # todo get new sample
    real_data = condition + ground_truth # todo change implementation
    generated_data = G(condition)
    discriminator_real_out = self.discriminator(real_data)
    discriminator_fake_out = self.discriminator(generated_data)
    loss_g, _ = get_losses(discriminator_real_out, discriminator_fake_out)
    
    #train discriminator
    condition, ground_truth = dataset.get_sample # todo get new sample
    real_data = condition + ground_truth # todo change implementation
    discriminator_real_out = self.discriminator(real_data)
    discriminator_fake_out = self.discriminator(generated_data)
    loss_g, _ = get_losses(discriminator_real_out, discriminator_fake_out)
 ```










