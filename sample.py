"""
Sample from a trained model
"""

import glob
import os
import pickle
from contextlib import nullcontext
from pathlib import Path

import tiktoken
import torch

from dag_logger import DAGLogger
from models.dag_model import GPT, GPTConfig
from python_version_check import check_python_version

check_python_version()

# -----------------------------------------------------------------------------
DEFAULT_SAMPLE_PROMPT = "Two plus 5 = "

init_from = "resume"
start = DEFAULT_SAMPLE_PROMPT
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = False
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# model
if init_from == "resume":
    checkpoint_dir = (
        "/runpod-volume/checkpoints"
        if os.path.exists("/runpod-volume")
        else "checkpoints"
    )

    checkpoint_files = glob.glob(str(Path(checkpoint_dir) / "ckpt_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    def get_iter_num(filename):
        return int(Path(filename).stem.split("_")[1])

    latest_checkpoint = max(checkpoint_files, key=get_iter_num)
    print(f"Loading checkpoint from: {latest_checkpoint}")

    checkpoint = torch.load(latest_checkpoint, map_location=device)

    model_args = checkpoint["model_args"]
    if "dag_depth" in model_args:
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    else:
        model_args.setdefault("dag_depth", 0)
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    init_from == "resume"
    and "config" in checkpoint
    and "dataset" in checkpoint["config"]
):
    meta_path = Path("data") / checkpoint["config"]["dataset"] / "meta.pkl"
    load_meta = meta_path.exists()
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
dag_logger = DAGLogger()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))

            if hasattr(model, "config") and model.config.dag_depth > 0:
                print("\nDAG Information:")
                dag_logger.compute_log_statistics(model)
                dag_logger.format_console_logging(model, decode_fn=decode, input_ids=x)

            print("---------------")
