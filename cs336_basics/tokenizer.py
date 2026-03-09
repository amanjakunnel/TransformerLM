import regex as re
from collections import Counter

PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{M}+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        parts = re.split(f"({special_pattern})", text)
    else:
        parts = [text]
    pre_tokens = []
    for part in parts:
        if part not in special_tokens:
            pre_tokens.extend(re.findall(PAT, part))
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    word_counts = Counter(tuple(bytes([b]) for b in t.encode("utf-8")) for t in pre_tokens)
    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in word_counts.items():
            for i in range(len(word) - 1):
                pairs[word[i], word[i+1]] += freq
        if not pairs:
            break
        best_pair = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]
        new_token = best_pair[0] + best_pair[1]
        new_id = len(vocab)
        vocab[new_id] = new_token
        merges.append(best_pair)
        new_word_counts = Counter()
        for word, freq in word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] = freq
        word_counts = new_word_counts
    return vocab, merges

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab 
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merges = {pair: self.byte_to_id[pair[0] + pair[1]] for pair in merges}
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile("|".join(re.escape(s) for s in sorted_specials))
        else:
            self.special_pattern = None
    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    def encode(self, text):
        if not self.special_pattern:
            parts = [text]
        else:
            parts = []
            last_end = 0
            for match in self.special_pattern.finditer(text):
                if match.start() > last_end:
                    parts.append(text[last_end:match.start()])
                parts.append(match.group())
                last_end = match.end()
            if last_end < len(text):
                parts.append(text[last_end:])
        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.byte_to_id[part.encode("utf-8")])
            else:
                pre_tokens = re.findall(PAT, part)
                for token in pre_tokens:
                    word = [bytes([b]) for b in token.encode("utf-8")]
                    while len(word) >= 2:
                        candidates = []
                        for i in range(len(word) - 1):
                            pair = (word[i], word[i+1])
                            if pair in self.merges:
                                candidates.append((list(self.merges.keys()).index(pair), i, pair))
                        if not candidates:
                            break
                        _, best_idx, best_pair = min(candidates)
                        new_word = word[:best_idx] + [best_pair[0] + best_pair[1]] + word[best_idx+2:]
                        word = new_word
                    ids.extend(self.byte_to_id[b] for b in word)
        return ids
    def decode(self, ids):
        byte_stream = b"".join(self.vocab[idx] for idx in ids)
        return byte_stream.decode("utf-8", errors="replace")