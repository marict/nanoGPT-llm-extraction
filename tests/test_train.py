import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def test_train_script_runs(tmp_path):
    prep_script = os.path.join(REPO_ROOT, 'data', 'shakespeare_char', 'prepare.py')
    subprocess.check_call([sys.executable, prep_script])

    train_script = os.path.join(REPO_ROOT, 'train.py')
    config_file = os.path.join(REPO_ROOT, 'config', 'train_default.py')

    cmd = [
        sys.executable,
        train_script,
        config_file,
        f'--out_dir={tmp_path}',
        '--device=cpu',
        '--compile=False',
        '--eval_interval=1',
        '--eval_iters=1',
        '--log_interval=1',
        '--max_iters=0',
         '--dataset=shakespeare_char',
        '--batch_size=2',
        '--n_layer=1',
        '--n_head=1',
        '--n_embd=32',
        '--block_size=32',
    ]
    subprocess.check_call(cmd, cwd=REPO_ROOT)
