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


'''