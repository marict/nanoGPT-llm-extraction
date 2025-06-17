"""Evaluation config for a DAGGPT checkpoint on RunPod."""

batch_size = 8
eval_iters = 500
eval_only = True
wandb_log = False
init_from = "resume"

dag_depth = 4
dag_hidden_dim = 16
