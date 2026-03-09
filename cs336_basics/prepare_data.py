import numpy as np
import pickle
import os
from cs336_basics.tokenizer import Tokenizer, train_bpe

VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]
TRAIN_TEXT = "data/TinyStoriesV2-GPT4-train.txt"
VALID_TEXT = "data/TinyStoriesV2-GPT4-valid.txt"

def prepare_data():
    vocab_path = "cs336_basics/vocab.pkl"
    merges_path = "cs336_basics/merges.pkl"
    if not os.path.exists(vocab_path):
        print(f"BPE files not found. Training on {TRAIN_TEXT}...")
        if not os.path.exists(TRAIN_TEXT):
             raise FileNotFoundError(f"Could not find {TRAIN_TEXT}. Is it in the data folder?")
        vocab, merges = train_bpe(TRAIN_TEXT, VOCAB_SIZE, SPECIAL_TOKENS)
        with open(vocab_path, "wb") as f: pickle.dump(vocab, f)
        with open(merges_path, "wb") as f: pickle.dump(merges, f)
    else:
        with open(vocab_path, "rb") as f: vocab = pickle.load(f)
        with open(merges_path, "rb") as f: merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, special_tokens=SPECIAL_TOKENS)
    os.makedirs("cs336_basics/data_bin", exist_ok=True)
    for name, path in [("train", TRAIN_TEXT), ("valid", VALID_TEXT)]:
        print(f"Tokenizing {name} split from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            ids = list(tokenizer.encode_iterable(f))
        out_path = f"cs336_basics/data_bin/{name}.bin"
        np.array(ids, dtype=np.uint16).tofile(out_path)
        print(f"Saved {len(ids)} tokens to {out_path}")

if __name__ == "__main__":
    prepare_data()