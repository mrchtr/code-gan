"""
Class for source code generation.
model = MyModelDefinition(args)
optimizer = optim.SGD(model.parameters(﻿)﻿, lr=﻿0.001﻿, momentum=﻿0.9﻿)
﻿
checkpoint = torch.load(﻿'load/from/path/model.pth'﻿)
model.load_state_dict(checkpoint[﻿'model_state_dict'﻿]﻿)
optimizer.load_state_dict(checkpoint[﻿'optimizer_state_dict'﻿]﻿)
epoch = checkpoint[﻿'epoch'﻿]
loss = checkpoint[﻿'loss'﻿]


"""
import torch
import wandb
from transformers import GPT2Tokenizer

from config import init_config
from models.generator.TransformerGenerator import PretrainedGPTGenerator
from utils.Preprocessor import postprocess

if __name__ == '__main__':
    #input = "class OrderManager():<EOL><INDENT>self.init():<EOL><INDENT>if self.order_manager is None : "
    #input = "class OrderManager():<EOL><INDENT>"
    input = "import pandas as pd <EOL> def request(url, method):<EOL> "

    config = init_config()
    logger = wandb.init(project=config.project_name, config=config)

    print("Init tokenizer and model ...")

    tokenizer = GPT2Tokenizer(vocab_file="code-tokenizer-vocab.json", merges_file="code-tokenizer-merges.txt")
    tokenizer.add_tokens(config.special_tokens)
    config.vocab_size = len(tokenizer)
    config.eos_token_id = tokenizer.encode("<EOL>")[0]
    config.pad_token_id = tokenizer.encode("<pad>")[0]

    config.vocab_size = len(tokenizer)
    generator = PretrainedGPTGenerator(config, bos_token=config.eos_token_id)
    artifact = logger.use_artifact(config.saved_model, type='model')
    artifact_dir = artifact.download()
    generator.load_state_dict(torch.load(artifact_dir + '/generator.pth', map_location=torch.device(config.device)))

    # context, sequence_length, batch_size, num_samples=1, min_len=0, forward_gumbel=True
    context = tokenizer.encode(input, return_tensors='pt')
    max_sequence_len = 128
    min_sequence_len = 0
    batch_size = 1
    num_samples = 1
    forward_gumbel = False
    generated = generator.gen_sample(context, max_sequence_len, batch_size, num_samples, min_sequence_len, forward_gumbel)

    # decode
    generated = tokenizer.decode(generated[0], skip_special_tokens=False)

    # postprocess
    generated = postprocess(generated)

    # output
    print(generated)




