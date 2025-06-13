import argparse
import os
import torch
import tiktoken
from model import GPT, GPTConfig
from dag_model import DAGGPT, DAGGPTConfig


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt["model_args"]
    if "dag_depth" in model_args:
        cfg = DAGGPTConfig(**model_args)
        model = DAGGPT(cfg)
    else:
        cfg = GPTConfig(**model_args)
        model = GPT(cfg)
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q, a = line.split("\t")
            q = q.replace("Q:", "").strip()
            a = a.replace("A:", "").strip()
            pairs.append((q, a))
    return pairs


def evaluate(model, enc, pairs, device):
    losses = []
    for q, a in pairs:
        ids = enc.encode(q + " " + a)
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None, :]
        y = torch.tensor(ids[1:], dtype=torch.long, device=device)[None, :]
        with torch.no_grad():
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description="Compare two checkpoints")
    parser.add_argument("--ckpt_baseline", required=True)
    parser.add_argument("--ckpt_dag", required=True)
    parser.add_argument("--dataset", default="tests/math_eval.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    pairs = load_pairs(args.dataset)

    baseline = load_model(args.ckpt_baseline, device)
    dag = load_model(args.ckpt_dag, device)

    base_loss = evaluate(baseline, enc, pairs, device)
    dag_loss = evaluate(dag, enc, pairs, device)

    print(f"Baseline loss: {base_loss:.4f}")
    print(f"DAG loss: {dag_loss:.4f}")


if __name__ == "__main__":
    main()
