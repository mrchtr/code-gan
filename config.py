import wandb
import torch


def init_config():
    config = wandb.config

    """
    If debug is true, a small part of the dataset will be procceded instead of the whole.
    To run in production mode, please the debug to false.
    Furthermore, no metrics will be log into wandb if debug is true.  
    """
    config.debug = False

    # project name in wandb
    config.project_name = "code-gan"
    config.saved_model = 'mrchtr/code-gan/model:v2'

    # hardware settings
    config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config.data_dir = "./data/dataset"
    config.training_data = "./data/dataset/out_train.txt"
    config.validation_data = "./data/dataset/out_test.txt"

    # architecture settings

    # config.vocab_size = 52000 # is initialized during the start of process
    config.special_tokens = [
        '<BOF>',  # Begin of File
        '<EOF>',  # End of File
        '<EOL>',  # End of Line
        '<COMMENT>',  # Comment
        '<STR_LIT>',  # String literal > 15 chars
        '<INT_LIT>',  # Integer literal
        '<INDENT>',  # Indent
        '<DEDENT>',  # Dedent
    ]


    """
    Sequence length settings. 
        - start_sequence_len : size of the start sequence that is used as condition for the generation during GAN training
        - sequence_len : length of generated sequence including the condition
        - block_size : the dataset is splitted into blocks for the training process. block_size = start_len + seq_len 
    """
    config.start_sequence_len = 128
    config.sequence_length = 128  # 75
    config.block_size = config.sequence_length

    """
    Different generator that could be used inside the GAN architecture.
        - GPT : Transformer based architecture with gumbel-softmax linear layer on top
    """
    config.generator = "GPT"

    """
    Different discriminator that could be used inside the GAN architecture.
        - CNN : CNN based architecure based on the idea of https://openreview.net/forum?id=rJedV3R5tm
        - CodeBERT: TODO
    """
    config.discriminator = "CNN"
    config.discriminator_embedding_dim = 64

    """
    Training parameters
    """

    config.batch_size = 64

    # Pretraining
    config.pretrain_optimizer = "AdamW"
    config.lr_pretrain = 5e-5
    config.pretraining_epochs = 0

    # GAN training
    config.generator_optimizer = "Adam"
    config.discriminator_optimizer = "Adam"
    config.lr_adv_g = 1e-4  # 1e-4
    config.lr_adv_d = 1e-4  # 1e-4
    config.nadv_steps = 1000
    config.open_end_generation = True
    config.g_steps = 1
    config.d_steps = 5
    config.temperature = 50
    config.loss_type = "wgan-gp"  # standard, rsgan, wgan or wgan-gp
    config.clip_norm = 2
    config.freezing_discriminator = False  # freeze layers of pretrained discriminator ?
    config.freezing_generator = False  # freeze layers of the pretrained generator ?
    config.repetition_penalty = 1.2

    return config