"""Tokenize every turn of ShareGPT_V3_unfiltered_cleaned_split.json with the Llama-3 tokenizer.

Output: a parquet/csv with one row per turn: conv_idx, turn_idx, role, num_tokens.
Downstream scripts derive request (prompt, decode) lengths from this.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import tiktoken
from tiktoken.load import load_tiktoken_bpe

PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

_enc = None


def _init(model_path):
    global _enc
    ranks = load_tiktoken_bpe(model_path)
    _enc = tiktoken.Encoding(name="llama3", pat_str=PAT, mergeable_ranks=ranks, special_tokens={})


def _count(batch):
    out = []
    for conv_idx, turns in batch:
        for t_idx, t in enumerate(turns):
            n = len(_enc.encode_ordinary(t["value"]))
            out.append((conv_idx, t_idx, t["from"], n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--tokenizer", required=True, help="Llama-3 tokenizer.model (tiktoken BPE)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    data = json.load(open(args.sharegpt))
    items = [(i, c["conversations"]) for i, c in enumerate(data)]
    chunks = [items[i : i + 2000] for i in range(0, len(items), 2000)]
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.tokenizer,)) as ex:
        for r in ex.map(_count, chunks):
            rows.extend(r)
    df = pd.DataFrame(rows, columns=["conv_idx", "turn_idx", "role", "num_tokens"])
    df.to_csv(args.out, index=False)
    print(f"{len(data)} conversations, {len(df)} turns -> {args.out}")


if __name__ == "__main__":
    main()
