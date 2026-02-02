import re;

class SimpleTokeniserV1:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab;
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text: str):
        preprocessed: list[str] = re.split(r'([,.:;?_!"()\']|--|\s)', text);
        preprocessed = [item for item in preprocessed if item.strip()];
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ];
        ids: list[int] = [self.str_to_int[s] for s in preprocessed];
        return ids;

    def decode(self, ids: list[int]):
        text: str = " ".join([self.int_to_str[i] for i in ids]);
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text);
        return text;