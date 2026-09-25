"""Turn per-turn ShareGPT token counts into (num_prefill_tokens, num_decode_tokens) tables.

first_turn : vLLM-benchmark style -- first human message is the prompt, first
             assistant reply is the decode. (One request per conversation.)
multi_turn : every assistant reply is a request whose prompt is the full
             conversation history (Sarathi/Vidur "sharegpt_8k" style).
Both keep requests with >=4 prompt and decode tokens and total <= 8192.
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_total", type=int, default=8192)
    args = ap.parse_args()

    t = pd.read_csv(args.turns).sort_values(["conv_idx", "turn_idx"])
    os.makedirs(args.out_dir, exist_ok=True)

    first = t[t.turn_idx < 2].pivot(index="conv_idx", columns="turn_idx", values=["role", "num_tokens"]).dropna()
    ok = (first[("role", 0)] == "human") & (first[("role", 1)] == "gpt")
    ft = pd.DataFrame({
        "num_prefill_tokens": first.loc[ok, ("num_tokens", 0)].astype(int).values,
        "num_decode_tokens": first.loc[ok, ("num_tokens", 1)].astype(int).values,
    })

    t["hist"] = t.groupby("conv_idx").num_tokens.cumsum() - t.num_tokens
    r = t[(t.role == "gpt") & (t.turn_idx > 0)]
    mt = pd.DataFrame({"num_prefill_tokens": r["hist"].values, "num_decode_tokens": r["num_tokens"].values})

    for name, df in [("first_turn", ft), ("multi_turn", mt)]:
        df = df[(df.num_prefill_tokens >= 4) & (df.num_decode_tokens >= 4)
                & (df.num_prefill_tokens + df.num_decode_tokens <= args.max_total)]
        path = os.path.join(args.out_dir, f"sharegpt_llama3_{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name}: {len(df)} requests, mean prefill {df.num_prefill_tokens.mean():.0f}, "
              f"mean decode {df.num_decode_tokens.mean():.0f} -> {path}")


if __name__ == "__main__":
    main()
