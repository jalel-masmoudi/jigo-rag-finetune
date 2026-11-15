"""Run the GPU steps in order.

Do the CPU steps first, on any machine:

    python make_data.py
    python generate_qa.py
    python auto_filter_test.py

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
    training = [command for command in STEPS if command[-1] != "--suite"]
    suite = [sys.executable, "eval.py", "--suite"]
    failed = []
    for command in training:
        print("running", " ".join(command), flush=True)
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError as exc:
            failed.append(f"{' '.join(command)} (exit {exc.returncode})")
            print("that step failed. Finished results are kept. Continuing.", flush=True)
    print("running", " ".join(suite), flush=True)
    suite_code = subprocess.call(suite)
    if suite_code != 0:
        failed.append(f"{' '.join(suite)} (exit {suite_code})")
    if failed:
        raise SystemExit(
            "Some steps failed:\n"
            + "\n".join(failed)
            + "\nEval ran afterward for whatever training finished. Read results/."
        )
    print("Done. Read results/*.json. Leave the README table empty until those files are from data/test_reviewed.jsonl.")


if __name__ == "__main__":
    main()
