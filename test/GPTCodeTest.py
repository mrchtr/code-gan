from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from torch import nn
from data.Dataset import TextDataset
import torch.nn.functional as F


def decode(tokens):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))


training_data = "../demo_code/out_train.txt"
block_size=16

tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2")
model = AutoModelWithLMHead.from_pretrained("microsoft/CodeGPT-small-py-adaptedGPT2")

with open(training_data) as f:
    content = "".join(f.readlines())

dataset = TextDataset(inp=tokenizer.encode(content), block_size=block_size)

x, y = dataset.__getitem__(0)
print(f"X: {x}")
print(f"Y: {y} --> same as x shifted one to the right")
print(f"Back to text again: {tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y))}")


criterion = nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
train_loader = DataLoader(dataset, batch_size=2, drop_last=True)

model.train()

loss_fct = CrossEntropyLoss()

for batch, (x, y) in enumerate(train_loader):
    output = model(x)


    # test for getting next possible token
    logits = model(x).logits # last layer logits of batch : 0
    logits = F.gumbel_softmax(logits)
    next_token = [torch.argmax(logits[0], dim=-1)[-1]]
    decoded = decode(next_token)

    #### loss calculation ####
    # Shift so that tokens < n predict n
    shift_logits = output.logits[..., :-1, :].contiguous()  # remove the last logits in every batch
    shift_labels = y[..., 1:].contiguous()  # removing the first tokens in each label sequence
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss.backward()
    optimizer.step()
    #print(f"next token: {decoded}")
    #print(f"whole sequence: {decode(x[0])} -:- {decoded}")
    print(f"should be :: {decode([y[0][-1]])} - is :: {decoded}")
    print(f"loss: {loss.item()}")
