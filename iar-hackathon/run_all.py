"""Run the full pipeline: embed -> experiments -> figures."""
import os
import subprocess
import sys

from seed import set_seed


def run_embed():
    """Run embed.py to compute embeddings (only if not cached)."""
    emb_path = "embeddings.npy"
    if os.path.exists(emb_path):
        print("embeddings.npy exists, skipping embedding step.")
        print("  NOTE: if this file predates the embed.py fix (normalisation +"
              " resize), delete it and re-run so good embeddings are generated.")
        return
    print("Running embedding step (fixed: ImageNet normalise + resize + probe gate)...")
    subprocess.run([sys.executable, "embed.py"], check=True)


def run_experiments():
    """Run experiments.py to generate results.csv."""
    print("Running experiments...")
    subprocess.run([sys.executable, "experiments.py"], check=True)


def run_figures():
    """Run figures.py to generate plots."""
    os.makedirs("results/figures", exist_ok=True)
    print("Generating figures...")
    subprocess.run([sys.executable, "figures.py"], check=True)


def main():
    set_seed(42)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/matrices", exist_ok=True)

    run_embed()
    run_experiments()
    run_figures()
    print("Done. Results are in results/.")


if __name__ == "__main__":
    main()
