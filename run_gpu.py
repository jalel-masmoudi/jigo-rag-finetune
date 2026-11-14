"""Run the GPU steps in order.

Do the CPU steps first, on any machine:

    python make_data.py
    python generate_qa.py
    python review_test.py

Then, on a free Kaggle or Colab GPU:

    pip install -r requirements.txt
    pip install bitsandbytes
    python run_gpu.py

bitsandbytes is only needed for the 4-bit generator step. The script
stops if CUDA is missing. It writes JSON under results/. It does not
edit the README. Copy the numbers into the results table yourself,
and only after you have opened the JSON and checked "smoke" is false.
"""

import subprocess
import sys

STEPS = (
    [sys.executable, "eval.py"],
    [sys.executable, "train_embed.py"],
    [sys.executable, "train_reranker.py"],
    [sys.executable, "train_lora.py"],
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
