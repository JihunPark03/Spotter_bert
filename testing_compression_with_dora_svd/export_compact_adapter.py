import argparse
import csv
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export physically compact SVD-pruned LoRA factors."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Experiment directories or a results root containing experiment directories.",
    )
    parser.add_argument(
        "--output_name",
        default="compact_svd_adapter",
        help="Subdirectory name written inside each experiment directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing compact export directories.",
    )
    return parser.parse_args()


def find_experiment_dirs(paths):
    experiment_dirs = []
    for raw_path in paths:
        path = Path(raw_path)
        if (path / "adapter_model.safetensors").exists():
            experiment_dirs.append(path)
            continue

        if not path.exists():
            raise FileNotFoundError(path)

        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "adapter_model.safetensors").exists():
                experiment_dirs.append(child)

    return experiment_dirs


def read_effective_ranks(experiment_dir):
    rank_path = experiment_dir / "effective_rank_per_layer.csv"
    if not rank_path.exists():
        return {}

    ranks = {}
    with rank_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            ranks[row["layer"]] = int(row["effective_rank"])
    return ranks


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def export_compact_adapter(experiment_dir, output_name, overwrite=False):
    adapter_path = experiment_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    ranks = read_effective_ranks(experiment_dir)
    if not ranks:
        raise FileNotFoundError(
            f"{experiment_dir} has no effective_rank_per_layer.csv; "
            "only SVD-pruned runs can be compact-exported."
        )

    output_dir = experiment_dir / output_name
    if output_dir.exists() and not overwrite:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = load_file(str(adapter_path))
    compact_state = {}
    layer_rows = []
    original_lora_params = 0
    compact_lora_params = 0
    original_lora_bytes = 0
    compact_lora_bytes = 0

    handled = set()
    for key, tensor in state_dict.items():
        if not key.endswith(".lora_A.weight"):
            continue

        layer = key[: -len(".lora_A.weight")]
        a_key = key
        b_key = f"{layer}.lora_B.weight"
        if b_key not in state_dict:
            raise KeyError(f"Missing matching LoRA B tensor for {a_key}")

        lora_a = state_dict[a_key]
        lora_b = state_dict[b_key]
        kept_rank = ranks.get(layer)
        if kept_rank is None:
            raise KeyError(f"Missing effective rank for layer {layer}")

        max_rank = min(lora_a.shape[0], lora_b.shape[1])
        kept_rank = max(0, min(int(kept_rank), int(max_rank)))
        compact_a = lora_a[:kept_rank, :].contiguous()
        compact_b = lora_b[:, :kept_rank].contiguous()

        compact_state[a_key] = compact_a
        compact_state[b_key] = compact_b
        handled.add(a_key)
        handled.add(b_key)

        original_params = int(lora_a.numel() + lora_b.numel())
        compact_params = int(compact_a.numel() + compact_b.numel())
        original_bytes = tensor_bytes(lora_a) + tensor_bytes(lora_b)
        compact_bytes = tensor_bytes(compact_a) + tensor_bytes(compact_b)
        original_lora_params += original_params
        compact_lora_params += compact_params
        original_lora_bytes += original_bytes
        compact_lora_bytes += compact_bytes

        layer_rows.append(
            {
                "layer": layer,
                "original_rank": int(max_rank),
                "compact_rank": kept_rank,
                "a_shape": list(lora_a.shape),
                "b_shape": list(lora_b.shape),
                "compact_a_shape": list(compact_a.shape),
                "compact_b_shape": list(compact_b.shape),
                "original_lora_params": original_params,
                "compact_lora_params": compact_params,
            }
        )

    for key, tensor in state_dict.items():
        if key in handled:
            continue
        # DoRA magnitude vectors, classifier weights, and other non-LoRA tensors
        # are not pruned. They are copied exactly into the compact artifact.
        compact_state[key] = tensor

    compact_path = output_dir / "adapter_model.safetensors"
    save_file(compact_state, str(compact_path))

    non_lora_params = sum(
        int(tensor.numel())
        for key, tensor in state_dict.items()
        if ".lora_A.weight" not in key and ".lora_B.weight" not in key
    )
    non_lora_bytes = sum(
        tensor_bytes(tensor)
        for key, tensor in state_dict.items()
        if ".lora_A.weight" not in key and ".lora_B.weight" not in key
    )

    manifest = {
        "format": "compact_svd_adapter_v1",
        "source_adapter": str(adapter_path),
        "compact_adapter": str(compact_path),
        "peft_loadable": False,
        "note": (
            "LoRA A/B tensors are stored with physical per-layer compact ranks. "
            "DoRA magnitude and classifier tensors are copied unchanged. This "
            "artifact is for storage/parameter accounting and custom compact "
            "loading, not direct PEFT loading."
        ),
        "num_layers": len(layer_rows),
        "original_lora_params": original_lora_params,
        "compact_lora_params": compact_lora_params,
        "non_lora_params": non_lora_params,
        "original_total_params_in_adapter": original_lora_params + non_lora_params,
        "compact_total_params_in_adapter": compact_lora_params + non_lora_params,
        "lora_param_reduction_ratio": (
            1.0 - compact_lora_params / original_lora_params
            if original_lora_params
            else 0.0
        ),
        "adapter_param_reduction_ratio": (
            1.0
            - (compact_lora_params + non_lora_params)
            / (original_lora_params + non_lora_params)
            if original_lora_params + non_lora_params
            else 0.0
        ),
        "original_lora_bytes": original_lora_bytes,
        "compact_lora_bytes": compact_lora_bytes,
        "non_lora_bytes": non_lora_bytes,
        "original_file_bytes": adapter_path.stat().st_size,
        "compact_file_bytes": compact_path.stat().st_size,
        "file_size_reduction_ratio": (
            1.0 - compact_path.stat().st_size / adapter_path.stat().st_size
            if adapter_path.stat().st_size
            else 0.0
        ),
        "layers": layer_rows,
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    with (output_dir / "layers.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    return manifest


def main():
    args = parse_args()
    experiment_dirs = find_experiment_dirs(args.paths)
    exported = []
    skipped = []

    for experiment_dir in experiment_dirs:
        try:
            manifest = export_compact_adapter(
                experiment_dir,
                output_name=args.output_name,
                overwrite=args.overwrite,
            )
        except FileNotFoundError:
            skipped.append(experiment_dir)
            continue

        if manifest is None:
            skipped.append(experiment_dir)
            continue

        exported.append((experiment_dir, manifest))
        print(
            f"exported {experiment_dir}: "
            f"file_reduction={manifest['file_size_reduction_ratio']:.3f}, "
            f"adapter_param_reduction={manifest['adapter_param_reduction_ratio']:.3f}, "
            f"lora_param_reduction={manifest['lora_param_reduction_ratio']:.3f}"
        )

    if exported:
        common_parent = Path(args.paths[0])
        if not common_parent.is_dir() or (common_parent / "adapter_model.safetensors").exists():
            common_parent = exported[0][0].parent

        summary_path = common_parent / "compact_export_summary.csv"
        fieldnames = [
            "experiment",
            "compact_adapter",
            "original_file_bytes",
            "compact_file_bytes",
            "file_size_reduction_ratio",
            "original_total_params_in_adapter",
            "compact_total_params_in_adapter",
            "adapter_param_reduction_ratio",
            "original_lora_params",
            "compact_lora_params",
            "lora_param_reduction_ratio",
            "non_lora_params",
            "num_layers",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for experiment_dir, manifest in exported:
                writer.writerow(
                    {
                        "experiment": experiment_dir.name,
                        **{key: manifest[key] for key in fieldnames if key != "experiment"},
                    }
                )
        print(f"Wrote {summary_path}")

    print(f"Exported {len(exported)} compact adapters; skipped {len(skipped)}.")


if __name__ == "__main__":
    main()
