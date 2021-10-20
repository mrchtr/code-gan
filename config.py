import wandb
import torch


def init_config():
    config = wandb.config

    """
    If debug is true, a small part of the dataset will be procceded instead of the whole.
    To run in production mode, please the debug to false.
    Furthermore, no metrics will be log into wandb if debug is true.  
    """
    config.debug = True

    # project name in wandb
    config.project_name = "code-gan"

    # hardware settings
    config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # dataset configuration
    config.benchmark_dataset = False  # True or False
    # config.training_data = "./demo_code/out_train.txt"
    config.data_dir = "./data/dataset"
    config.training_data = "./data/dataset/out_train.txt"
    config.validation_data = "./data/dataset/out_test.txt"

    # tokenizer configuration
    #config.vocab_size = 52000
    config.special_tokens = [
        '<BOF>',
        '<EOF>',
        '<EOL>',
        '<COMMENT>',
        '<STR_LIT>',
        '<INT_LIT>',
        '<INDENT>',
        '<DEDENT>',
    ]


    config.block_size = 128  # in case of LSTM / Memory Unit should be 1


    # generator model
    config.generator = "GPTCode"  # LSTM,  Transformer or GPTCode

    #pretraining generator
    config.pretrain_generator = True

    # --- used for Transformer XL
    config.ninp = 768  # default: 768
    config.nhead = 12
    config.nhid = 200  # default LSTM: 128 / Transformer: 200
    config.nlayers = 12  # default LSTM: 1 / Transformer 12
    config.embedding_dim = 8  # embedding vector size for LSTM
    config.dropout = 0.5


    # discriminator model
    config.discriminator = "CNN"
    config.discriminator_embedding_dim = 32



    # training related parameter & hyper parameters
    config.pretrain_optimizer = "AdamW"
    config.generator_optimizer = "Adam"
    config.discriminator_optimizer = "Adam"

    config.clip_norm = 2

    """
    size of the generated example sequences. 
    """
    config.sequence_length = 148 #75

    """
    size of the given context for the sequence generation
    """
    config.start_sequence_len = 20 #25
    config.batch_size = 64
    # config.pretraining_epochs = 0
    config.pretraining_steps = 5000
    config.lr_pretrain = 5e-5
    config.lr_adv_g = 1e-4  # 1e-4
    config.lr_adv_d = 1e-4  # 1e-4
    config.nadv_steps = 20000
    config.g_steps = 1
    config.d_steps = 5
    config.temperature = 50
    config.loss_type = "wgan-gp" #standard, rsgan, wgan or wgan-gp
    config.noise_as_context = False
    config.freezing = False
    return config