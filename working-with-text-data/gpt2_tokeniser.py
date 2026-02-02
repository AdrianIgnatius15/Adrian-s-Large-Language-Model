import importlib;
import tiktoken;

'''
## Byte-Pair Encoding
#### Using the open-source library from OpenAI called "Tiktoken"

A byte-pair encoder is like the encoding and decoding of a list of tokenised words,
same as what has been done in "SimpleTokeniserV1".
But this byte-pair encoder is much faster using the library called "Tiktoken" and it's
developed using Rust instead of Python although it exports it's methods/functions to Python.
'''

## Display Tiktoken Version.
print("Tiktoken version", tiktoken.__version__);

## Initialise tokeniser
tokeniser = tiktoken.get_encoding("gpt2");
text: str = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
);
integers = tokeniser.encode(text, allowed_special={"<|endoftext|>"});
print(integers);
strings = tokeniser.decode(integers);
print(strings);