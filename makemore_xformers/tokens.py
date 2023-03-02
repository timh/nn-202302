# %%
from typing import List, Dict, Tuple, Union
import re
from dataclasses import dataclass

import torch
from torch import Tensor

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
    
    def words_to_tokens(self, words: List[str]) -> List[int]:
        if self.include_special:
            res = [self.vocab_to_token.get(word, self.token_unk) for word in words]
            res = [self.token_start.token] + res + [self.token_end.token]
        else:
            res = [self.vocab_to_token[word] for word in words]
        return res
    
    def words_to_tensors(self, words: List[str], device = "cpu") -> List[Tensor]:
        tokens = self.words_to_tokens(words)
        return torch.tensor(tokens, dtype=torch.long, device=device)
    
    def tokens_to_words(self, tokens: Union[List[int], List[Tensor]]) -> List[str]:
        if len(tokens) == 0:
            return list()

        if isinstance(tokens[0], Tensor):
            tokens = [t.item() for t in tokens]
        return [self.token_to_vocab[token] for token in tokens]

    def tokens_to_str(self, tokens: Union[List[int], List[Tensor]]) -> str:
        return "".join(self.tokens_to_words(tokens))


class TextReader:
    seq_len: int
    inputs: List[Tensor]
    truth: List[Tensor]

    dictionary: Dictionary

    def __init__(self, seq_len: int, tokenizer: Tokenizer, filename: str, include_special: bool, device = "cpu"):
        import datetime
        # read all text
        with open(filename, "r") as file:
            text = file.read()
            all_words = tokenizer.tokenize(text)

        self.tokenizer = tokenizer
        self.dictionary = Dictionary(all_words, include_special=include_special)

        all_tokens = self.dictionary.words_to_tensors(all_words, device=device)

        nexamples = len(all_tokens) - seq_len - 1
        if include_special:
            nexamples += 1 # <sot>
        self.inputs = list()
        self.truth = list()

        for i in range(nexamples):
            self.inputs.append(all_tokens[i:i + seq_len])
            self.truth.append(all_tokens[i + 1:i + seq_len + 1])

        self.seq_len = seq_len

        print(f"TextReader: {seq_len=} {self.dictionary.vocab_len=}")
    
    def as_pairs(self) -> List[Tuple[Tensor, Tensor]]:
        return list(zip(self.inputs, self.truth))

class WordTextReader(TextReader):
    def __init__(self, seq_len: int, wordlen: int, filename: str, include_special: bool, device="cpu"):
        tokenizer = WordTokenizer(wordlen=wordlen)
        super().__init__(seq_len=seq_len, tokenizer=tokenizer, filename=filename, include_special=include_special, device=device)

# %%
# wtr = WordTextReader(64, 2, "shakespeare-1000.txt", include_special=False, device="cuda")
# first = wtr.as_pairs()[0][0]
# last = wtr.as_pairs()[-1][0]

# print("first:", wtr.dictionary.tokens_to_str(first))
# print("last:", wtr.dictionary.tokens_to_str(last))

# print("first:", "|".join(wtr.dictionary.tokens_to_words(first)))

# %%

