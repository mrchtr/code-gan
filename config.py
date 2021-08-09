import wandb
import torch


def init_config():
    config = wandb.config

    # hardware settings
    config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




    # generator model
    config.generator = "GPTCode"  # LSTM,  Transformer or GPTCode

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
    config.benchmark_dataset = False  # True or False
    #config.training_data = "./demo_code/out_train.txt"
    config.training_data = "./demo_code/out_train.txt"
    config.validation_data = "./demo_code/out_test.txt"
    config.data_dir = "./demo_code"
    config.block_size = 128  # in case of LSTM / Memory Unit should be 1
    config.vocab_size = 32000
    config.special_tokens = [
        '<BOF>',
        '<EOF>',
        '<COMMENT>',
        '<STR_LIT>',
        '<INT_LIT>'
    ]

    # training related parameter & hyper parameters
    config.pretrain_optimizer = "AdamW"
    config.generator_optimizer = "AdamW"
    config.discriminator_optimizer = "AdamW"

    config.sequence_length = 20  # size of generated examples
    config.start_sequence_len = 5
    config.batch_size = 2
    # config.pretraining_epochs = 0
    config.pretraining_steps = 1
    config.lr_pretrain = 5e-5
    config.lr_adv_g = 1e-4  # 1e-4
    config.lr_adv_d = 1e-4  # 1e-4
    config.nadv_steps = 20000
    config.g_steps = 1
    config.d_steps = 1
    config.temperature = 50
    config.loss_type = "rsgan"
    config.noise_as_context = False

    return config
