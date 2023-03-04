# %%
from typing import List, Dict, Tuple, Union
import re
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

class Tokenizer:
    def tokenize(text: str) -> List[str]:
        pass

class RegexTokenizer(Tokenizer):
    regexes: List[re.Pattern[str]]

    def __init__(self, regexes: List[re.Pattern[str]]):
        self.regexes = regexes
    
    def tokenize(self, text: str) -> List[str]:
        res: List[str] = list()
        while len(text) > 0:
            matched = False
            for regex in self.regexes:
                match = regex.match(text)
                if match:
                    res.append(match.group(0))
                    text = text[match.end():]
                    matched = True
                    break

            if not matched:
                print(f"rest of string not matched: {text=}")
                break
        return res

# class WordTokenizer(RegexTokenizer):
#     def __init__(self, wordlen: int):
#         if wordlen:
#             wlstr = "{1," + str(wordlen) + "}"
#         else:
#             wlstr = ""
#         pats: List[re.Pattern[str]] = list()
#         pats.append(" ")
#         pats.append("\n")
#         pats.append(r"\d")
#         pats.append(f"\\w{wlstr}")
#         pats.append(r"[^\w\d]")
#         pats = [re.compile(p) for p in pats]
#         super().__init__(pats)

RE_WORD = re.compile("(\w+)|(\d)|([^\w\d])")
class WordTokenizer(Tokenizer):
    def __init__(self, wordlen: int):
        self.wordlen = wordlen
    
    def tokenize(self, text: str) -> List[str]:
        res: List[str] = list()
        lines = text.split("\n")
        for lineidx, line in enumerate(lines):
            words = line.split(" ")
            for wordidx, word in enumerate(words):
                while len(word) > 0:
                    match = RE_WORD.match(word)
                    if match:
                        if self.wordlen:
                            res.append(match.group(0)[:self.wordlen])
                            word = match.group(0)[self.wordlen:] + word[match.end():]
                        else:
                            res.append(match.group(0))
                            word = word[match.end():]
                    else:
                        res.append(word)
                        break
                if wordidx != len(words) - 1:
                    res.append(" ")
            if lineidx != len(lines) - 1:
                res.append("\n")
        return res
                        

@dataclass
class SpecialToken:
    text: str
    token: int
    key: object = object()

    def __hash__(self) -> int:
        return hash(self.key)

class Dictionary:
    vocab_len: int
    vocab_to_token: Dict[str, int]
    token_to_vocab: Dict[int, str]
    
    token_start = SpecialToken("<sot>", 0)
    token_end = SpecialToken("<eot>", 1)
    token_unk = SpecialToken("<unk>", 2)

    def __init__(self, all_words: List[str], include_special: bool):
        special_offset = 3 if include_special else 0
        uniq_strs = sorted(list(set(all_words)))
        self.vocab_to_token = {word: i + special_offset for i, word in enumerate(uniq_strs)}
        self.token_to_vocab = {i + special_offset: word for i, word in enumerate(uniq_strs)}
        self.vocab_len = len(uniq_strs) + special_offset

        self.include_special = include_special
        if self.include_special:
            for i, st in enumerate([self.token_start, self.token_end, self.token_unk]):
                self.vocab_to_token[st] = i
                self.token_to_vocab[i] = st.text
    
    def words_to_tokens(self, words: List[str], include_startend = False) -> List[int]:
        if self.include_special:
            res = [self.vocab_to_token.get(word, self.token_unk) for word in words]
            if include_startend:
                res = [self.token_start.token] + res + [self.token_end.token]
        else:
            res = [self.vocab_to_token[word] for word in words]
        return res
    
    def words_to_tensors(self, words: List[str], include_startend = False, device = "cpu", dtype = torch.long) -> List[Tensor]:
        tokens = self.words_to_tokens(words, include_startend=include_startend)
        return torch.tensor(tokens, device=device, dtype=dtype)
    
    def tokens_to_words(self, tokens: Union[List[int], List[Tensor]]) -> List[str]:
        if len(tokens) == 0:
            return list()

        if isinstance(tokens[0], Tensor):
            tokens = [t.item() for t in tokens]
        return [self.token_to_vocab[token] for token in tokens]

    """
                    tensor:  (batch, seqlen, vocablen)
        .softmax / argmax -> (batch, seqlen)
    """
    def probs_to_str(self, tensor: Tensor) -> str:
        norm_probs = torch.nn.functional.softmax(tensor, dim=-1)
        tokens = torch.argmax(norm_probs, dim=-1)
        return self.tokens_to_str(tokens)

    def tokens_to_str(self, tokens: Union[List[int], List[Tensor]]) -> str:
        return "".join(self.tokens_to_words(tokens))

class TextReader: pass
@dataclass
class _TextReaderIter:
    _start: int
    _end: int
    treader: TextReader
    _idx: int = 0

    def __next__(self) -> Tuple[Tensor, Tensor]:
        start = self._idx + self._start
        end = start + self.treader.seq_len

        res = (self.treader.inputs[start:end], self.treader.inputs[start + 1:end + 1])
        self._idx += 1
        return res
    
    def __len__(self) -> int:
        return self._end - self._start
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]:
        if isinstance(idx, slice):
            res: List[Tuple[Tensor, Tensor]] = list()
            for i in idx.indices(len(self)):
                start = self._start + i
                end = start + self.treader.seq_len
                inputs = self.treader.all_tokens[start:end]
                truth = self.treader.all_tokens[start + 1:end + 1]
                res.append((inputs, truth))
            return res

        start = self._start + idx
        end = start + self.treader.seq_len
        return self.treader.all_tokens[start:end], self.treader.all_tokens[start + 1:end + 1]
        

class TextReader(Dataset):
    seq_len: int
    inputs: List[Tensor]
    truth: List[Tensor]

    dictionary: Dictionary

    def __init__(self, seq_len: int, tokenizer: Tokenizer, filename: str, include_special: bool, device = "cpu"):
        import datetime
        # read all text
        with open(filename, "r") as file:
            text = file.read()
            start = datetime.datetime.now()
            all_words = tokenizer.tokenize(text)
            end = datetime.datetime.now()
            print(f"- {end - start} to tokenize {filename}")

        self.tokenizer = tokenizer
        self.dictionary = Dictionary(all_words, include_special=include_special)

        start = datetime.datetime.now()
        all_tokens = self.dictionary.words_to_tensors(all_words, include_startend=True, device=device)
        end = datetime.datetime.now()
        print(f"- {end - start} to call words_to_tensors")

        self.nexamples = len(all_tokens) - seq_len - 1
        if include_special:
            self.nexamples += 1 # <sot>
        self.seq_len = seq_len
        self.all_tokens = all_tokens

        print(f"TextReader: {seq_len=} {self.dictionary.vocab_len=} {self.nexamples=}")

    def __len__(self) -> int:
        return self.nexamples
    
    def __iter__(self):
        return _TextReaderIter(_start=0, _end=self.nexamples, treader=self)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[Tensor, Tensor], List[Tuple[Tensor, Tensor]]]:
        return _TextReaderIter(_start=0, _end=self.nexamples, treader=self)[idx]

    def train_val_split(self, idx: int) -> Tuple[_TextReaderIter, _TextReaderIter]:
        train_split = _TextReaderIter(0, idx, treader=self)
        val_split = _TextReaderIter(idx, self.nexamples, self)
        return train_split, val_split

class WordTextReader(TextReader):
    def __init__(self, seq_len: int, wordlen: int, filename: str, include_special: bool, device="cpu"):
        tokenizer = WordTokenizer(wordlen=wordlen)
        super().__init__(seq_len=seq_len, tokenizer=tokenizer, filename=filename, include_special=include_special, device=device)

# %%
# wtr = WordTextReader(1024, 1, "all_python_100000.txt", include_special=True, device="cuda")
# first = wtr.as_pairs()[0][0]
# last = wtr.as_pairs()[-1][0]

# print("first:", wtr.dictionary.tokens_to_str(first))
# print("last:", wtr.dictionary.tokens_to_str(last))

# print("first:", "|".join(wtr.dictionary.tokens_to_words(first)))

# %%

