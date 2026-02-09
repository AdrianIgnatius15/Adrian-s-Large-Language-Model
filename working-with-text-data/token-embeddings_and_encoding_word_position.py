'''
## Creating Token Embeddings

Token embeddings are creation of map of words which then helps the machine
to see relationships between and creating semantic with them.

This helps in predicting accurately the next words from an input by understanding
the semantic of the relationship.
'''
from adrian_llm import create_dataloader_v1;

import os;
import requests;#
import torch;

if not os.path.exists("the-verdict.txt"):
    url: str = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt");
    file_path: str = "the-verdict.txt";

    response = requests.get(url, timeout=30);
    response.raise_for_status();
    with open(file_path, "wb") as f:
        f.write(response.content);

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text: str = f.read();

dataloader, vocab_size = create_dataloader_v1(raw_text, batch_size = 1, max_length = 1, stride = 1, shuffle = False);
data_iter = iter(dataloader);
first_batch = next(data_iter);
second_batch = next(data_iter);

## Let's take a sample tokens which have converted the words into an array to demonstrate token embeddings.
input_ids = torch.tensor([2, 3, 5, 1]);
## Then, let's create a embedding layer to create token embeddings using "Tensor".
#### It take the length of the tokenised array which is `tokeniser.n_vocab` function & the desired size of the vector.
# embedding_layer = torch.nn.Embedding(50257, )
output_dim: int = 256; ## The size of the vector, it like having columns in a matrix multiplication.

torch.manual_seed(123) ## Since we are using neural networks, it will assign random weights. Hence we introduce manually to the tokens manually the weights.
embedding_layer = torch.nn.Embedding(vocab_size, output_dim);
print("Weight matrix", embedding_layer.weight);
# print("Show vector number 3 in the vector array", embedding_layer(torch.tensor([3])));

'''
## Encoding word positions

Encoding word positions are important as sometimes certain words in the token after token embedding may appear again.
To make use of that occurence we add more information to it and the process of it is called "Encoding word positions".
Certain LLM model like "GPT2" implements a second token embedding layer.

Instead of focusing on the absolute position of a token, the emphasis of relative positional embeddings is on the relative
position or distance between tokens. This means the model learns the relationships in terms of "how far apart" rather than
"at which exact position". The advantage here is that the model can generalise better to sequences of varying lengths.
'''
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(token_embedding_layer.shape)

max_length: int = 4
dataloader_2, vocab_size_2 = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
);
data_iter = iter(dataloader_2);
inputs, targets = next(data_iter);

print("Token IDs: \n", inputs);
print("\n Inputs shape: \n", inputs.shape);
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
## GPT-2 uses absolute position embeddings, so we just create another embedding layer
context_length: int = max_length;
position_embedding_layer = torch.nn.Embedding(context_length, output_dim);
print(position_embedding_layer.weight);
position_embeddings = position_embedding_layer(torch.arange(max_length));
print("Position embeddings", position_embeddings);
## To create the input embeddings used in LLM, we simply add the token and the positional embeddings:
## token_embeddings + position_embeddings
input_embeddings = token_embeddings + position_embeddings;
print(input_embeddings.shape);