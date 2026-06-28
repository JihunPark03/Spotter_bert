import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_MODEL_DIRS = [
    "modernbert-large-fake-review-detector-full",
    "modernbert-large-fake-review-detector-lora",
    "modernbert-large-fake-review-detector-adalora",
    "modernbert-large-fake-review-detector-starlora",
]


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def round_value(value, digits=4):
    if value is None:
        return ""

    if isinstance(value, float):
        return round(value, digits)

    return value


def find_latest_trainer_state(model_dir):
    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )

    for checkpoint in reversed(checkpoints):
        state_path = checkpoint / "trainer_state.json"
        if state_path.exists():
            return state_path

    return None


def best_eval_from_trainer_state(model_dir):
    state_path = find_latest_trainer_state(model_dir)
    if state_path is None:
        return {}

    state = read_json(state_path)
    best_checkpoint = state.get("best_model_checkpoint")
    best_metric = state.get("best_metric")
    best_step = ""
    best_epoch = ""
    best_loss = None

    for item in state.get("log_history", []):
        item_f1 = item.get("eval_f1") or item.get("eval_macro_f1")
        if item_f1 is None:
            continue

        if best_metric is not None and abs(float(item_f1) - float(best_metric)) > 1e-12:
            continue

        best_step = item.get("step", "")
        best_epoch = item.get("epoch", "")
        best_loss = item.get("eval_loss")
        break

    return {
        "best_checkpoint": best_checkpoint or "",
        "best_step": best_step,
        "best_epoch": best_epoch,
        "best_eval_f1": best_metric,
        "best_eval_loss": best_loss,
    }


def summarize_model(model_dir):
    info_path = model_dir / "experiment_info.json"
    if not info_path.exists():
        return {
            "method": model_dir.name,
            "status": "missing experiment_info.json",
            "model_dir": str(model_dir),
        }

    info = read_json(info_path)
    train_metrics = info.get("train_metrics", {})
    eval_metrics = info.get("eval_metrics", info.get("metrics", {}))
    best_eval = best_eval_from_trainer_state(model_dir)

    return {
        "method": info.get("method", model_dir.name),
        "status": "ok",
        "num_samples": info.get("num_samples", infer_num_samples(model_dir)),
        "train_samples": info.get("train_samples", ""),
        "eval_samples": info.get("eval_samples", ""),
        "rank": info.get("rank", ""),
        "init_rank": info.get("init_rank", ""),
        "lora_alpha": info.get("lora_alpha", ""),
        "lora_dropout": info.get("lora_dropout", ""),
        "final_eval_f1": round_value(
            eval_metrics.get("eval_f1") or eval_metrics.get("eval_macro_f1")
        ),
        "final_eval_loss": round_value(eval_metrics.get("eval_loss")),
        "best_eval_f1": round_value(best_eval.get("best_eval_f1")),
        "best_eval_loss": round_value(best_eval.get("best_eval_loss")),
        "best_step": best_eval.get("best_step", ""),
        "best_epoch": best_eval.get("best_epoch", ""),
        "train_loss": round_value(train_metrics.get("train_loss")),
        "train_runtime_sec": round_value(train_metrics.get("train_runtime"), digits=2),
        "train_samples_per_second": round_value(
            train_metrics.get("train_samples_per_second"), digits=2
        ),
        "train_steps_per_second": round_value(
            train_metrics.get("train_steps_per_second"), digits=2
        ),
        "best_checkpoint": best_eval.get("best_checkpoint", ""),
        "model_dir": str(model_dir),
    }


def infer_num_samples(model_dir):
    match = re.search(r"_n(\d+)(?:$|_)", model_dir.name)
    if match:
        return int(match.group(1))
    return ""


def discover_model_dirs(root):
    return sorted(
        path.relative_to(root)
        for path in root.iterdir()
        if path.is_dir() and (path / "experiment_info.json").exists()
    )


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, fieldnames):
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    return "\n".join(lines)


def write_markdown(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(rows, fieldnames) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize fine-tuning experiment_info.json files into final tables."
    )
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing model output folders.",
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).resolve().parent / "final_comparison",
        type=Path,
        help="Output path prefix or directory. Writes CSV and Markdown files.",
    )
    parser.add_argument(
        "model_dirs",
        nargs="*",
        default=None,
        help="Model output directories relative to --root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dirs = args.model_dirs or DEFAULT_MODEL_DIRS
    if not args.model_dirs:
        discovered = discover_model_dirs(args.root)
        if discovered:
            model_dirs = discovered

    rows = [summarize_model(args.root / model_dir) for model_dir in model_dirs]
    fieldnames = [
        "method",
        "status",
        "num_samples",
        "train_samples",
        "eval_samples",
        "rank",
        "init_rank",
        "lora_alpha",
        "lora_dropout",
        "final_eval_f1",
        "final_eval_loss",
        "best_eval_f1",
        "best_eval_loss",
        "best_step",
        "best_epoch",
        "train_loss",
        "train_runtime_sec",
        "train_samples_per_second",
        "train_steps_per_second",
        "best_checkpoint",
        "model_dir",
    ]

    output = args.output
    csv_path = output if output.suffix == ".csv" else output.with_suffix(".csv")
    md_path = output if output.suffix == ".md" else output.with_suffix(".md")

    write_csv(csv_path, rows, fieldnames)
    write_markdown(md_path, rows, fieldnames)

    print(markdown_table(rows, fieldnames[:16]))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
