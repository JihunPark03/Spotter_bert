import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import TrainerCallback


@dataclass
class SVDPruningConfig:
    start_step: int = 0
    start_ratio: float | None = None
    interval: int = 100
    threshold: float = 0.0
    keep_ratio: float = 1.0
    energy_ratio: float | None = None
    mode: str = "periodic"
    min_rank: int = 1
    max_rank: int | None = None
    adapter_name: str = "default"


@dataclass
class LayerPruningRecord:
    step: int
    layer: str
    original_rank: int
    kept_rank: int
    threshold_rank: int
    keep_ratio_rank: int
    energy_rank: int | None = None
    singular_values_before: list[float] = field(default_factory=list)
    singular_values_after: list[float] = field(default_factory=list)


def _get_adapter_module(module_dict, adapter_name):
    if hasattr(module_dict, "__contains__") and adapter_name in module_dict:
        return module_dict[adapter_name]

    if hasattr(module_dict, "keys"):
        keys = list(module_dict.keys())
        if keys:
            return module_dict[keys[0]]

    return None


def iter_lora_direction_layers(model, adapter_name="default"):
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        lora_a = _get_adapter_module(module.lora_A, adapter_name)
        lora_b = _get_adapter_module(module.lora_B, adapter_name)

        if lora_a is None or lora_b is None:
            continue

        if not hasattr(lora_a, "weight") or not hasattr(lora_b, "weight"):
            continue

        yield name, lora_a.weight, lora_b.weight


def _rank_from_threshold(singular_values, threshold):
    if threshold <= 0:
        return int(singular_values.numel())

    return int((singular_values > threshold).sum().item())


def _choose_rank(singular_values, config, max_factor_rank=None):
    original_rank = int(singular_values.numel())
    threshold_rank = _rank_from_threshold(singular_values, config.threshold)
    keep_ratio_rank = max(1, int(math.ceil(original_rank * config.keep_ratio)))
    energy_rank = None

    if config.energy_ratio is not None:
        if not 0 < config.energy_ratio <= 1:
            raise ValueError("energy_ratio must be in (0, 1].")
        energy = singular_values.square()
        total_energy = energy.sum()
        if float(total_energy.item()) == 0:
            energy_rank = config.min_rank
        else:
            cumulative = torch.cumsum(energy, dim=0) / total_energy
            energy_rank = int((cumulative < config.energy_ratio).sum().item()) + 1
        rank = energy_rank
    elif config.threshold > 0:
        rank = min(threshold_rank, keep_ratio_rank)
    else:
        rank = keep_ratio_rank

    rank = max(config.min_rank, rank)

    if config.max_rank is not None:
        rank = min(config.max_rank, rank)

    if max_factor_rank is not None:
        rank = min(max_factor_rank, rank)
        threshold_rank = min(max_factor_rank, threshold_rank)
        keep_ratio_rank = min(max_factor_rank, keep_ratio_rank)
        if energy_rank is not None:
            energy_rank = min(max_factor_rank, energy_rank)

    return min(original_rank, max(0, rank)), threshold_rank, keep_ratio_rank, energy_rank


@torch.no_grad()
def prune_lora_direction_with_svd(model, config):
    records = []

    for layer_name, lora_a_weight, lora_b_weight in iter_lora_direction_layers(
        model,
        adapter_name=config.adapter_name,
    ):
        original_dtype = lora_a_weight.dtype
        device = lora_a_weight.device
        lora_a = lora_a_weight.detach().float()
        lora_b = lora_b_weight.detach().float()

        # SVD pruning is applied only to the LoRA direction update:
        # Delta_W = B @ A. DoRA magnitude parameters are not read or modified.
        delta_w = lora_b @ lora_a
        u, singular_values, vh = torch.linalg.svd(delta_w, full_matrices=False)
        factor_rank = min(
            int(lora_a.shape[0]),
            int(lora_b.shape[1]),
            int(singular_values.numel()),
        )
        candidate_singular_values = singular_values[:factor_rank]
        kept_rank, threshold_rank, keep_ratio_rank, energy_rank = _choose_rank(
            candidate_singular_values,
            config,
            max_factor_rank=factor_rank,
        )

        pruned_singular_values = torch.zeros_like(singular_values)
        if kept_rank > 0:
            pruned_singular_values[:kept_rank] = singular_values[:kept_rank]

        new_a = torch.zeros_like(lora_a)
        new_b = torch.zeros_like(lora_b)

        if kept_rank > 0:
            sqrt_s = singular_values[:kept_rank].sqrt()
            new_a[:kept_rank, :] = sqrt_s[:, None] * vh[:kept_rank, :]
            new_b[:, :kept_rank] = u[:, :kept_rank] * sqrt_s[None, :]

        lora_a_weight.copy_(new_a.to(device=device, dtype=original_dtype))
        lora_b_weight.copy_(new_b.to(device=device, dtype=original_dtype))

        records.append(
            LayerPruningRecord(
                step=-1,
                layer=layer_name,
                original_rank=factor_rank,
                kept_rank=kept_rank,
                threshold_rank=threshold_rank,
                keep_ratio_rank=keep_ratio_rank,
                energy_rank=energy_rank,
                singular_values_before=[float(value) for value in singular_values.cpu()],
                singular_values_after=[float(value) for value in pruned_singular_values.cpu()],
            )
        )

    return records


def write_pruning_records(output_dir, records):
    if not records:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "svd_pruning_log.jsonl"
    csv_path = output_dir / "effective_rank_per_layer.csv"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.__dict__) + "\n")

    latest_by_layer = {}
    for record in records:
        latest_by_layer[record.layer] = record

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "layer",
                "original_rank",
                "effective_rank",
                "threshold_rank",
                "keep_ratio_rank",
                "energy_rank",
            ],
        )
        writer.writeheader()
        for record in latest_by_layer.values():
            writer.writerow(
                {
                    "step": record.step,
                    "layer": record.layer,
                    "original_rank": record.original_rank,
                    "effective_rank": record.kept_rank,
                    "threshold_rank": record.threshold_rank,
                    "keep_ratio_rank": record.keep_ratio_rank,
                    "energy_rank": record.energy_rank,
                }
            )


class DoraSVDPruningCallback(TrainerCallback):
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = Path(output_dir)
        self.records = []
        self._last_pruned_step = None
        self._resolved_start_step = config.start_step

    def _resolve_start_step(self, state):
        if self.config.start_ratio is None:
            return self.config.start_step
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps <= 0:
            return self.config.start_step
        return max(1, int(math.ceil(max_steps * self.config.start_ratio)))

    def _should_prune(self, step, state):
        self._resolved_start_step = self._resolve_start_step(state)
        if step <= 0:
            return False
        if step < self._resolved_start_step:
            return False
        if self._last_pruned_step == step:
            return False
        if self.config.interval <= 0:
            return False
        return (step - self._resolved_start_step) % self.config.interval == 0

    def _write_records(self):
        write_pruning_records(self.output_dir, self.records)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if model is None or not self._should_prune(step, state):
            return control

        records = prune_lora_direction_with_svd(model, self.config)
        for record in records:
            record.step = step
        self.records.extend(records)
        self._last_pruned_step = step
        self._write_records()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.records:
            self._write_records()
        return control


def estimate_effective_direction_params(model, adapter_name="default"):
    total = 0
    ranks = {}

    for layer_name, lora_a_weight, lora_b_weight in iter_lora_direction_layers(
        model,
        adapter_name=adapter_name,
    ):
        with torch.no_grad():
            delta_w = lora_b_weight.detach().float() @ lora_a_weight.detach().float()
            singular_values = torch.linalg.svdvals(delta_w)
            factor_rank = min(
                int(lora_a_weight.shape[0]),
                int(lora_b_weight.shape[1]),
                int(singular_values.numel()),
            )
            candidate_singular_values = singular_values[:factor_rank]
            if candidate_singular_values.numel() == 0:
                effective_rank = 0
            else:
                max_singular_value = float(candidate_singular_values.max().item())
                tolerance = max(1e-8, max_singular_value * 1e-5)
                effective_rank = int((candidate_singular_values > tolerance).sum().item())
            total += effective_rank * (
                int(lora_a_weight.shape[1]) + int(lora_b_weight.shape[0])
            )
            ranks[layer_name] = effective_rank

    return total, ranks
