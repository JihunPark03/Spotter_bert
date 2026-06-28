from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "testing" / "training"
FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"

COLORS = {
    "full": "#4C78A8",
    "lora": "#F58518",
    "adalora": "#54A24B",
    "star_lora": "#B279A2",
}

LABELS = {
    "full": "Full",
    "lora": "LoRA",
    "adalora": "AdaLoRA",
    "star_lora": "Star-LoRA",
}

METHOD_ORDER = ["full", "lora", "adalora", "star_lora"]


def style_axis(ax):
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def final_summary(final_df):
    df = final_df.set_index("method").loc[METHOD_ORDER].reset_index()
    x = range(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].bar(
        x,
        df["final_eval_f1"],
        color=[COLORS[m] for m in df["method"]],
        width=0.62,
    )
    axes[0].set_ylim(0.99, 1.0)
    axes[0].set_title("Final F1")
    axes[0].set_ylabel("F1")
    for index, value in enumerate(df["final_eval_f1"]):
        axes[0].text(index, value + 0.00015, f"{value:.4f}", ha="center", fontsize=8)
    style_axis(axes[0])

    axes[1].bar(
        x,
        df["final_eval_loss"],
        color=[COLORS[m] for m in df["method"]],
        width=0.62,
    )
    axes[1].set_title("Final Eval Loss")
    axes[1].set_ylabel("loss")
    for index, value in enumerate(df["final_eval_loss"]):
        axes[1].text(index, value + 0.00055, f"{value:.4f}", ha="center", fontsize=8)
    style_axis(axes[1])

    axes[2].bar(
        x,
        df["train_runtime_sec"] / 60,
        color=[COLORS[m] for m in df["method"]],
        width=0.62,
    )
    axes[2].set_title("Training Runtime")
    axes[2].set_ylabel("minutes")
    for index, value in enumerate(df["train_runtime_sec"] / 60):
        axes[2].text(index, value + 0.5, f"{value:.1f}", ha="center", fontsize=8)
    style_axis(axes[2])

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABELS[m] for m in df["method"]], rotation=20)

    fig.suptitle("Full-Data Experiment: ModernBERT Fake Review Detection", y=1.04)
    save(fig, "final_summary.png")


def datasize_f1(datasize_df):
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    for method in METHOD_ORDER:
        sub = datasize_df[datasize_df["method"] == method].sort_values("num_samples")
        ax.plot(
            sub["num_samples"],
            sub["final_eval_f1"],
            marker="o",
            linewidth=2.3 if method == "star_lora" else 1.8,
            markersize=5.5,
            color=COLORS[method],
            label=LABELS[method],
            alpha=1.0 if method == "star_lora" else 0.88,
        )

    ax.set_xscale("log")
    ax.set_ylim(0.2, 1.04)
    ax.set_xlabel("training samples, log scale")
    ax.set_ylabel("final evaluation F1")
    ax.set_title("Data-Size Sweep")
    ax.legend(ncol=4, frameon=False, loc="lower right")
    style_axis(ax)
    save(fig, "datasize_f1_logx.png")


def star_lora_delta(datasize_df):
    pivot = datasize_df.pivot(index="num_samples", columns="method", values="final_eval_f1")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.plot(
        pivot.index,
        pivot["star_lora"] - pivot["adalora"],
        marker="o",
        color=COLORS["adalora"],
        linewidth=2.2,
        label="Star-LoRA - AdaLoRA",
    )
    ax.plot(
        pivot.index,
        pivot["star_lora"] - pivot["lora"],
        marker="s",
        color=COLORS["lora"],
        linewidth=2.2,
        label="Star-LoRA - LoRA",
    )
    ax.set_xscale("log")
    ax.set_xlabel("training samples, log scale")
    ax.set_ylabel("F1 difference")
    ax.set_title("Where Star-LoRA Helps")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, "star_lora_delta.png")


def rank_policy(datasize_df):
    sub = datasize_df[datasize_df["method"] == "star_lora"].sort_values("num_samples")

    fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
    ax1.plot(
        sub["num_samples"],
        sub["rank"],
        marker="o",
        linewidth=2.5,
        color=COLORS["star_lora"],
        label="target rank",
    )
    ax1.plot(
        sub["num_samples"],
        sub["init_rank"],
        marker="s",
        linewidth=2.0,
        color="#72B7B2",
        label="initial rank",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("training samples, log scale")
    ax1.set_ylabel("rank")
    ax1.set_title("Star-LoRA Dataset-Aware Capacity")
    style_axis(ax1)

    ax2 = ax1.twinx()
    ax2.plot(
        sub["num_samples"],
        sub["lora_dropout"],
        marker="^",
        linewidth=2.0,
        color="#E45756",
        label="dropout",
    )
    ax2.set_ylabel("dropout")
    ax2.set_ylim(0.0, 0.18)
    ax2.spines["top"].set_visible(False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper left")
    save(fig, "rank_policy.png")


def stability_by_layer():
    stability_path = (
        TRAINING_DIR
        / "modernbert-large-fake-review-detector-starlora"
        / "stability_scores.csv"
    )
    if not stability_path.exists():
        return

    df = pd.read_csv(stability_path, usecols=["step", "parameter_name", "stable_importance"])
    final_step = df["step"].max()
    final_df = df[df["step"] == final_step].copy()
    final_df["layer"] = final_df["parameter_name"].str.extract(r"layers\.(\d+)").astype(int)
    final_df["module"] = final_df["parameter_name"].str.extract(r"attn\.(Wqkv|Wo)")

    pivot = final_df.pivot_table(
        index="layer",
        columns="module",
        values="stable_importance",
        aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = pivot.index.to_numpy()
    width = 0.38
    ax.bar(x - width / 2, pivot["Wqkv"], width=width, color="#4C78A8", label="Wqkv")
    ax.bar(x + width / 2, pivot["Wo"], width=width, color="#72B7B2", label="Wo")
    ax.set_xlabel("ModernBERT layer")
    ax.set_ylabel("mean stable importance")
    ax.set_title(f"Star-LoRA Stability Monitor at Final Step {final_step}")
    ax.legend(frameon=False)
    ax.set_xticks(x)
    style_axis(ax)
    save(fig, "stability_by_layer.png")


def main():
    final_df = pd.read_csv(TRAINING_DIR / "final_comparison.csv")
    datasize_df = pd.read_csv(TRAINING_DIR / "datasize_comparison.csv")

    final_summary(final_df)
    datasize_f1(datasize_df)
    star_lora_delta(datasize_df)
    rank_policy(datasize_df)
    stability_by_layer()


if __name__ == "__main__":
    main()
