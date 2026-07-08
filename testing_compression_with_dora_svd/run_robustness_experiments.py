import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent

METHODS = {
    "dora_r16": {
        "method": "dora",
        "rank": 16,
        "lora_alpha": 32,
        "keep_ratio": 1.0,
    },
    "dora_r12": {
        "method": "dora",
        "rank": 12,
        "lora_alpha": 24,
        "keep_ratio": 1.0,
    },
    "dora_svd_r16_post_energy090": {
        "method": "dora_svd_pruning",
        "rank": 16,
        "lora_alpha": 32,
        "keep_ratio": 1.0,
        "energy_ratio": 0.90,
        "pruning_mode": "post_training",
    },
    "dora_svd_r16_post_energy095": {
        "method": "dora_svd_pruning",
        "rank": 16,
        "lora_alpha": 32,
        "keep_ratio": 1.0,
        "energy_ratio": 0.95,
        "pruning_mode": "post_training",
    },
}

PRESETS = {
    "paper_main_energy": {
        "datasets": ["sst2", "imdb"],
        "noise_rates": [0.0, 0.2, 0.4],
        "sample_sizes": ["1000", "5000"],
        "methods": [
            "dora_r12",
            "dora_r16",
            "dora_svd_r16_post_energy090",
            "dora_svd_r16_post_energy095",
        ],
        "seeds": [13, 21, 42],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DoRA post-training direction SVD compression sweeps."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output_root", default="testing_dora_svd/paper_main_energy_results"
    )
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="paper_main_energy"
    )
    parser.add_argument("--datasets", nargs="+", choices=["sst2", "imdb"], default=None)
    parser.add_argument("--noise_rates", nargs="+", type=float, default=None)
    parser.add_argument("--sample_sizes", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-large")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--svd_pruning_start_step", type=int, default=100)
    parser.add_argument("--svd_pruning_interval", type=int, default=100)
    parser.add_argument("--svd_pruning_threshold", type=float, default=0.0)
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--max_rank", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--collect_only", action="store_true")
    return parser.parse_args()


def selected_values(args):
    preset = PRESETS[args.preset]
    return {
        "datasets": args.datasets or preset["datasets"],
        "noise_rates": args.noise_rates or preset["noise_rates"],
        "sample_sizes": args.sample_sizes or preset["sample_sizes"],
        "methods": args.methods or preset["methods"],
        "seeds": args.seeds or preset["seeds"],
    }


def noise_name(noise_rate):
    if noise_rate == 0:
        return "clean"
    return f"noise{int(round(noise_rate * 100)):02d}"


def experiment_name(dataset_key, noise_rate, sample_size, method_name, seed):
    return f"{dataset_key}_{noise_name(noise_rate)}_n{sample_size}_{method_name}_seed{seed}"


def iter_experiments(args):
    values = selected_values(args)
    for dataset_key in values["datasets"]:
        for noise_rate in values["noise_rates"]:
            for sample_size in values["sample_sizes"]:
                for method_name in values["methods"]:
                    for seed in values["seeds"]:
                        method = METHODS[method_name]
                        yield {
                            "dataset_key": dataset_key,
                            "noise_rate": noise_rate,
                            "sample_size": str(sample_size),
                            "method_name": method_name,
                            "method": method["method"],
                            "rank": method["rank"],
                            "lora_alpha": method["lora_alpha"],
                            "keep_ratio": method["keep_ratio"],
                            "energy_ratio": method.get("energy_ratio"),
                            "pruning_mode": method.get("pruning_mode", "periodic"),
                            "start_ratio": method.get("start_ratio"),
                            "interval": method.get("interval"),
                            "seed": seed,
                            "name": experiment_name(
                                dataset_key,
                                noise_rate,
                                str(sample_size),
                                method_name,
                                seed,
                            ),
                        }


def experiment_dir(output_root, experiment):
    return Path(output_root) / experiment["name"]


def build_command(args, experiment):
    output_dir = experiment_dir(args.output_root, experiment)
    command = [
        args.python,
        str(PROJECT_DIR / "train_dora_svd.py"),
        "--method",
        experiment["method"],
        "--dataset_key",
        experiment["dataset_key"],
        "--model_name",
        args.model_name,
        "--output_dir",
        str(output_dir),
        "--max_length",
        str(args.max_length),
        "--rank",
        str(experiment["rank"]),
        "--lora_alpha",
        str(experiment["lora_alpha"]),
        "--label_noise_rate",
        str(experiment["noise_rate"]),
        "--svd_pruning_keep_ratio",
        str(experiment["keep_ratio"]),
        "--svd_pruning_start_step",
        str(args.svd_pruning_start_step),
        "--svd_pruning_interval",
        str(experiment["interval"] or args.svd_pruning_interval),
        "--svd_pruning_threshold",
        str(args.svd_pruning_threshold),
        "--min_rank",
        str(args.min_rank),
        "--svd_pruning_mode",
        experiment["pruning_mode"],
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--max_steps",
        str(args.max_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--seed",
        str(experiment["seed"]),
        "--report_to",
        args.report_to,
        "--run_name",
        experiment["name"],
    ]

    if experiment["sample_size"] != "full":
        command.extend(["--max_samples", experiment["sample_size"]])
    if args.max_rank is not None:
        command.extend(["--max_rank", str(args.max_rank)])
    if experiment["start_ratio"] is not None:
        command.extend(["--svd_pruning_start_ratio", str(experiment["start_ratio"])])
    if experiment["energy_ratio"] is not None:
        command.extend(["--svd_pruning_energy_ratio", str(experiment["energy_ratio"])])
    if args.bf16:
        command.append("--bf16")
    if args.fp16:
        command.append("--fp16")
    return command


def run_experiments(args):
    experiments = list(iter_experiments(args))
    print(f"Planned runs: {len(experiments)}")
    for experiment in experiments:
        output_dir = experiment_dir(args.output_root, experiment)
        info_path = output_dir / "experiment_info.json"
        if args.skip_existing and info_path.exists():
            print(f"Skipping existing run: {info_path}")
            continue

        command = build_command(args, experiment)
        print("Running:", " ".join(command))
        if args.dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, check=False, cwd=REPO_ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"Experiment failed: {experiment['name']}")


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def average_pruned_rank(output_dir, fallback_rank):
    rank_path = Path(output_dir) / "effective_rank_per_layer.csv"
    if not rank_path.exists():
        return fallback_rank
    with rank_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return fallback_rank
    ranks = [int(row["effective_rank"]) for row in rows]
    return round(sum(ranks) / len(ranks), 4)


def flatten_result(args, experiment):
    output_dir = experiment_dir(args.output_root, experiment)
    info_path = output_dir / "experiment_info.json"
    if not info_path.exists():
        return {
            "experiment": experiment["name"],
            "status": "missing",
            **{key: experiment[key] for key in [
                "dataset_key", "noise_rate", "sample_size", "method_name",
                "method", "rank", "lora_alpha", "keep_ratio", "energy_ratio",
                "pruning_mode", "seed"
            ]},
        }

    info = load_json(info_path)
    eval_metrics = info.get("eval_metrics", {})
    pre_pruning_eval_metrics = info.get("pre_pruning_eval_metrics") or {}
    train_metrics = info.get("train_metrics", {})
    return {
        "experiment": experiment["name"],
        "status": "ok",
        "dataset_key": experiment["dataset_key"],
        "noise_rate": experiment["noise_rate"],
        "sample_size": experiment["sample_size"],
        "method_name": experiment["method_name"],
        "method": experiment["method"],
        "rank": experiment["rank"],
        "lora_alpha": experiment["lora_alpha"],
        "keep_ratio": experiment["keep_ratio"],
        "energy_ratio": experiment["energy_ratio"],
        "pruning_mode": experiment["pruning_mode"],
        "seed": experiment["seed"],
        "train_samples": info.get("train_samples"),
        "eval_samples": info.get("eval_samples"),
        "validation_loss": eval_metrics.get("eval_loss"),
        "accuracy": eval_metrics.get("eval_accuracy"),
        "f1": eval_metrics.get("eval_f1"),
        "macro_f1": eval_metrics.get("eval_macro_f1"),
        "pre_pruning_validation_loss": pre_pruning_eval_metrics.get("eval_loss"),
        "pre_pruning_accuracy": pre_pruning_eval_metrics.get("eval_accuracy"),
        "pre_pruning_f1": pre_pruning_eval_metrics.get("eval_f1"),
        "pre_pruning_macro_f1": pre_pruning_eval_metrics.get("eval_macro_f1"),
        "train_loss": train_metrics.get("train_loss"),
        "trainable_params": info.get("trainable_params"),
        "avg_effective_rank": average_pruned_rank(output_dir, experiment["rank"]),
        "training_time_seconds": info.get("training_time_seconds"),
        "gpu_max_memory_mb": info.get("gpu_max_memory_mb"),
        "output_dir": str(output_dir),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment", "status", "dataset_key", "noise_rate", "sample_size",
        "method_name", "method", "rank", "lora_alpha", "keep_ratio", "seed",
        "energy_ratio", "pruning_mode",
        "train_samples", "eval_samples", "validation_loss", "accuracy", "f1",
        "macro_f1", "pre_pruning_validation_loss", "pre_pruning_accuracy",
        "pre_pruning_f1", "pre_pruning_macro_f1", "train_loss",
        "trainable_params", "avg_effective_rank",
        "training_time_seconds", "gpu_max_memory_mb", "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def collect_results(args):
    rows = [flatten_result(args, experiment) for experiment in iter_experiments(args)]
    write_csv(Path(args.output_root) / "robustness_results.csv", rows)


def main():
    args = parse_args()
    if not args.collect_only:
        run_experiments(args)
    if not args.dry_run:
        collect_results(args)


if __name__ == "__main__":
    main()
