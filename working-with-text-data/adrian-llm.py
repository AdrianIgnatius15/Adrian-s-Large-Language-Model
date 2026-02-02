from simple_tokeniser import SimpleTokeniserV1;
from torch.utils.data import Dataset, DataLoader;

import os;
import requests;
import re;
import tiktoken;
import torch;

"""
=========================================================
Get the sample data to create LLM model
=========================================================

This is to get the initial text data which is unlabeled data for initial training later.
"""
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
# tokeniser: tiktoken.Encoding = tiktoken.get_encoding("gpt2");
# encoded_text: list[int] = tokeniser.encode(raw_text);
# print("Length of the tokenised words in the array:", len(encoded_text));

'''
## PyTorch
#### Library

PyTorch is used for deep learning framework today and it has also a data loader
and it has already datasets ready to be used.
We can create our dataset, a machine learning to the learn the patterns of the data,
but re-inventing the wheel is tedious.

First, we create a dataset which we did from scratch but this time using "PyTorch" library as our utility to create one
'''
## First create a dataset using PyTorch
class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokeniser: tiktoken.Encoding, max_length: int, stride: int) -> None:
        self.input_ids = [];
        self.target_ids = [];

        # Tokenise the entire text
        token_ids: list[int] = tokeniser.encode(text, allowed_special={"<|endoftext|>"});
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length + 1";

        # Same concept of how we use sliding window to chunck the book into overlapping sequences of max length
        # This is to predict the next word but this is not the actual algorithm, this is for demo just like the previous code snippet
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length];
            target_chunck = token_ids[i + 1 : i + max_length + 1];
            self.input_ids.append(torch.tensor(input_chunk));
            self.target_ids.append(torch.tensor(target_chunck));

    def __len__(self):
        return len(self.input_ids);

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index];

## Next we use the Dataset class we just created using "PyTorch" with the sliding window technique to predict the next word
## Again, sliding window technique is to demonstrate the predicition but it's not the actual algorithm for machine learning to predict.
def create_dataloader_v1(
    text: str, 
    batch_size: int = 4, 
    max_length: int = 256, 
    stride: int = 128, 
    shuffle: bool = True, 
    drop_last: bool = True,
    num_workers: int = 0):
    # Initialiser the tokeniser
    tokeniser: tiktoken.Encoding = tiktoken.get_encoding("gpt2");

    # Create the dataset
    dataset: GPTDatasetV1 = GPTDatasetV1(text, tokeniser, max_length, stride);

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    );

    return dataloader;
    
# Now that create_dataloader_v1 is defined, we can use it
dataloader = create_dataloader_v1(raw_text, batch_size = 1, max_length = 1, stride = 1, shuffle = False);
data_iter = iter(dataloader);
first_batch = next(data_iter);
second_batch = next(data_iter);
print("First batch", first_batch);
print("Second batch", second_batch);

'''
To allow prediction, we want the inputs and targets
Since we want the model to predict the next word, the targets are the inputs shifted by one position to the right.

For example
`x: [290, 4920, 2241, 287]` first fours elements in the tokenised array
`y:      [4920, 2241, 287, 257]` the next four elements in the tokenised array
'''

'''
## Tokeniser used made from scratch

The commented code below has the logic to tokenise text using
a custom-built tokeniser. Uncommented it to use if you want to.
'''
# preprocessed: list[str] = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text);
# preprocessed = [item for item in preprocessed if item.strip()];
# print(preprocessed[:30]);

# all_words: list[str] = sorted(set(preprocessed));
# vocab_size: int = len(all_words);
# all_words.extend(["<|endoftext|>", "<|unk|>"]);

# vocab = {
#     token: integer 
#     for integer, token in enumerate(all_words)
# } ## Creates a list of objects like {1, "hello"}

# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break;


# tokeniser: SimpleTokeniserV1 = SimpleTokeniserV1(vocab=vocab);
# ids: list[int] = tokeniser.encode(" <|endoftext|> ".join(("Hello, do you like tea?", "In the sunlit terraces of the palace.")));
# print("Tokenised IDs", ids);