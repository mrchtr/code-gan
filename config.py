import wandb
import torch


def init_config():
    config = wandb.config

    # hardware settings
    config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # generator model
    config.generator = "Transformer"  # LSTM or Transformer

    # --- used for Transformer XL
    config.ninp = 768  # default: 768
    config.nhead = 12
    config.nhid = 200  # default LSTM: 128 / Transformer: 200
    config.nlayers = 12  # default LSTM: 1 / Transformer 12
    config.embedding_dim = 8  # embedding vector size for LSTM
    config.dropout = 0.5


    # discriminator model
    config.discriminator = "CNN"
    config.discriminator_embedding_dim = 1

    # dataset configuration
    config.training_data = "./demo_code/out_jokes.py"
    config.validation_data = "./demo_code/out_test.txt"
    config.data_dir = "./demo_code"
    config.block_size = 32  # in case of LSTM / Memory Unit should be 1
    config.vocab_size = 5000
    config.special_tokens = [
        '<BOF>',
        '<EOF>',
        '<COMMENT>',
        '<STR_LIT>',
        '<INT_LIT>'
    ]

    # training related parameter & hyper parameters
    config.pretrain_optimizer = "SGD"
    config.generator_optimizer = "Adam"
    config.discriminator_optimizer = "Adam"

    config.sequence_length = 40  # size of generated examples
    config.batch_size = 32
    config.pretraining_epochs = 0
    config.lr_pretrain = 5.0
    config.lr_adv = 1e-4
    config.nadv_steps = 2000
    config.g_steps = 1
    config.d_steps = 1
    config.temperature = 100

    return config
