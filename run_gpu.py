"""Run the GPU steps in order.

Do the CPU steps first, on any machine:

    python make_data.py
    python generate_qa.py
    python review_test.py

Then, on a free Kaggle or Colab GPU, do not reinstall torch. Those
machines already ship torch with CUDA. This repo checked torch 2.15.0.
Installing requirements.txt does not replace it. bitsandbytes is only
needed for the 4-bit generator step.

    pip install -r requirements.txt bitsandbytes
    python run_gpu.py

A CPU machine is different: install requirements-cpu.txt first so
torch==2.15.0 comes from the CPU wheel index, then requirements.txt.
Do not do that on the GPU machine.

The script stops if CUDA is missing. It writes JSON under results/.
It does not edit the README. Copy the numbers into the results table
yourself, and only after you have opened the JSON and checked "smoke"
is false. `python eval.py --suite` writes every table row that has
weights on disk, including the full pipeline, in results/table.json.
"""

import subprocess
import sys

STEPS = (
    [sys.executable, "train_embed.py"],
    [sys.executable, "train_reranker.py"],
    [sys.executable, "train_lora.py"],
    # One command writes every table row that has weights on disk,
    # including the baseline and the full pipeline.
    [sys.executable, "eval.py", "--suite"],
)


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script is the GPU run, not the CPU smoke checks.")
    for command in STEPS:
        print("running", " ".join(command), flush=True)
        subprocess.check_call(command)
    print("Done. Read results/*.json. Leave the README table empty until those files are from the reviewed test set.")


if __name__ == "__main__":
    main()
