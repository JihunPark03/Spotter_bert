import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_METHODS = ["full", "lora", "adalora", "star_lora"]
DEFAULT_SAMPLE_SIZES = [100, 1000, 5000, 10000, 40000]
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full/LoRA/AdaLoRA/Star-LoRA experiments by data size."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "datasize_runs",
        help="Directory where model outputs are written.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=PROJECT_DIR / "datasize_configs",
        help="Directory where generated config files are written.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override training.wandb_project for all generated configs.",
    )
    return parser.parse_args()


def load_yaml(path):
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def run_name(method, sample_size):
    return f"{method}_n{sample_size}"


def build_config(base_config, method, sample_size, output_dir, wandb_project):
    config = copy.deepcopy(base_config)
    config.setdefault("data", {})
    config.setdefault("fine_tuning", {})
    config.setdefault("training", {})

    config["data"]["max_samples"] = sample_size
    config["fine_tuning"]["method"] = method
    config["training"]["output_dir"] = str(output_dir)
    config["training"]["run_name"] = f"modernbert-{method}-n{sample_size}"

    if wandb_project:
        config["training"]["wandb_project"] = wandb_project

    return config


def run_one(args, base_config, method, sample_size):
    name = run_name(method, sample_size)
    output_dir = args.output_root / name
    config_path = args.config_root / f"{name}.yaml"
    info_path = output_dir / "experiment_info.json"

    if args.skip_existing and info_path.exists():
        print(f"Skipping existing run: {info_path}")
        return

    config = build_config(
        base_config=base_config,
        method=method,
        sample_size=sample_size,
        output_dir=output_dir,
        wandb_project=args.wandb_project,
    )
    write_yaml(config_path, config)

    command = [args.python, str(PROJECT_DIR / "main.py")]
    env = os.environ.copy()
    env["TRAIN_CONFIG"] = str(config_path)

    print("Running:", " ".join(command), f"TRAIN_CONFIG={config_path}")

    if args.dry_run:
        return

    completed = subprocess.run(command, cwd=PROJECT_DIR, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Run failed: method={method}, sample_size={sample_size}")


def main():
    args = parse_args()
    base_config = load_yaml(args.config)

    for sample_size in args.sample_sizes:
        for method in args.methods:
            run_one(args, base_config, method, sample_size)


if __name__ == "__main__":
    main()
